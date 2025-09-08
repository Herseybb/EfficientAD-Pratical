#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import itertools
import os
import random
from tqdm import tqdm
from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
import matplotlib.pyplot as plt

from config import get_config

# Define some common colors for console output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# constants
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

# Default data preprocessing pipeline (resize + normalization)
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Autoencoder augmentation:
# Instead of pixel-perfect reconstruction, we apply small perturbations
# to encourage the AE to learn more stable, invariant features.
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))


def get_quantile1(epoch, total_epochs, q_start=0.80, q_end=0.95):
    """
    Warm-up quantile scheduler (linear increase).
    Args:
        epoch: current training epoch (starting from 0)
        total_epochs: total number of epochs
        q_start: initial quantile
        q_end: final quantile
    """
    progress = min(1.0, epoch / total_epochs)
    return q_start + (q_end - q_start) * progress

def get_quantile2(epoch, schedule=None):
    """
    Step-based quantile scheduler.
    Args:
        epoch: current epoch (starting from 0)
        schedule: list of tuples (start_epoch, end_epoch, quantile)

    Example:
    schedule = [
        (0, 0, 0.0),      # epoch=0 → quantile=0.0
        (1, 10, 0.9),     # epochs 1-10 → quantile=0.9
        (11, 20, 0.95),   # epochs 11-20 → quantile=0.95
        (21, 999, 0.99),  # epoch >=21 → quantile=0.99
    ]
    """
    if schedule is None:
        raise ValueError("Schedule must be provided.")

    for start, end, q in schedule:
        if start <= epoch <= end:
            return q

    # If not covered by any range, return the last quantile
    return schedule[-1][2]


def image_level_score(map_np, mode="p99", topk=1000):
    """
    Aggregate pixel-level anomaly map into a single image-level score.
    """
    flat = map_np.reshape(-1)
    if mode == "p99":
        return float(np.percentile(flat, 99))
    elif mode == "topk":
        k = min(topk, flat.size)
        idx = np.argpartition(flat, -k)[-k:]
        return float(np.mean(flat[idx]))
    elif mode == "max":
        return float(np.max(flat))
    else:  # mean
        return float(np.mean(flat))

def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_config()

    # create output dir
    train_output_dir = os.path.join(config.output_dir, 'trainings',)
    # test_output_dir = os.path.join(config.output_dir, 'anomaly_maps', 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    # os.makedirs(test_output_dir, exist_ok=True)


    # === Load training and test data ===
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(config.dataset_path, config.train_path),
        transform=transforms.Lambda(train_transform))
    test_set = ImageFolderWithPath(
        os.path.join(config.dataset_path, config.test_path))

    # mvtec dataset paper recommend 10% validation set
    train_size = int(0.9 * len(full_train_set))
    validation_size = len(full_train_set) - train_size
    rng = torch.Generator().manual_seed(seed)
    train_set, validation_set = torch.utils.data.random_split(full_train_set,
                                                        [train_size,
                                                        validation_size],
                                                        rng)

    # Paper recommends batch size=1 for stability
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=0, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=1)


    # === Create models (Teacher / Student / Autoencoder) ===
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise Exception()
    teacher = torch.load(config.weights, map_location='cpu', weights_only=False)
    # teacher.load_state_dict(torch.load(config.weights, map_location='cpu'))
    autoencoder = get_autoencoder(out_channels)
    
    if config.student_weights:
        student = torch.load(config.student_weights, map_location='cpu', weights_only=False)
    if config.ae_weights:
        autoencoder = torch.load(config.ae_weights, map_location='cpu', weights_only=False)

    # Freeze teacher, train student and autoencoder
    teacher.eval()
    student.train()
    autoencoder.train()
    # Quick sanity check: teacher output should be normalized (mean≈0, std≈1)
    # with torch.no_grad():
    #     x = torch.randn(1, 3, 224, 224)
    #     y = teacher(x)
    #     print(y.mean(), y.std())

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    
    # === Define optimizers and schedulers ===
    optimizer_student = torch.optim.Adam(itertools.chain(student.parameters()),
                                 lr=2e-4, weight_decay=1e-5)
    scheduler_student = torch.optim.lr_scheduler.StepLR(
        optimizer_student, step_size=2, gamma=0.9)
    
    optimizer_ae = torch.optim.Adam(itertools.chain(autoencoder.parameters()),
                                 lr=1e-3, weight_decay=1e-5)
    scheduler_ae = torch.optim.lr_scheduler.StepLR(
        optimizer_ae, step_size=2, gamma=0.9)
    
    
    warm_up_epochs = 2
    for epoch in range(config.epochs):
        student.train()
        autoencoder.train()
        
        st_loss = 0.0
        ae_loss = 0.0
        stae_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")
        for batch_idx, (image_st, image_ae) in enumerate(pbar):
            
            if on_gpu:
                image_st = image_st.cuda()
                image_ae = image_ae.cuda()
        
            # ===== Loss =====
            # === student - teacher ===
            with torch.no_grad():
                teacher_output_st = teacher(image_st)
                teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
            student_output_st = student(image_st)[:, :out_channels]
            distance_st = (teacher_output_st - student_output_st) ** 2
            
            # Hard-mining: only backpropagate on the hardest features
            # q = get_quantile1(epoch, config.epochs, q_start=0.9, q_end=0.95) # change constantly
            # q = 0.0 if epoch < warm_up_epochs else 0.95 # use warmup
            q = get_quantile2(epoch, schedule = [
                                (0, 0, 0.0),      
                                (1, 5, 0.9),  
                                (6, 10, 0.95),   
                                (11, 999, 0.99),
                                        ])
            d_hard = torch.quantile(distance_st, q=q) # 
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])
            loss_st = loss_hard

            
            # === student & autoencoder ===
            ae_output = autoencoder(image_ae)
            with torch.no_grad():
                teacher_output_ae = teacher(image_ae)
                teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
            student_output_ae = student(image_ae)[:, out_channels:]
            
            # Autoencoder should approximate teacher
            distance_ae = (teacher_output_ae - ae_output)**2
            loss_ae = torch.mean(distance_ae)
            
            # Student should mimic autoencoder (detached to avoid AE being updated)
            distance_stae = (ae_output.detach() - student_output_ae)**2  
            loss_stae = torch.mean(distance_stae)
            
            
            # === Gradient updates ===
            # Update Student
            loss_st_total = 2 * loss_st + loss_stae
            optimizer_student.zero_grad()
            loss_st_total.backward()
            optimizer_student.step()
            
            # Update Autoencoder
            optimizer_ae.zero_grad()
            loss_ae.backward()
            optimizer_ae.step()
            
            # Logging
            st_loss += loss_st.item()
            ae_loss += loss_ae.item()
            stae_loss += loss_stae.item()
            pbar.set_postfix({
            "loss_st": loss_st.item(),
            "loss_ae": loss_ae.item(),
            "loss_stae": loss_stae.item(),
            })
        
        # Update LR once per epoch
        scheduler_student.step()
        scheduler_ae.step()
        
        st_loss /= len(train_loader)
        ae_loss /= len(train_loader)
        stae_loss /= len(train_loader)
        
        # ===== Validation at epoch end =====
        student.eval()
        autoencoder.eval()

        # Compute normalization bounds using validation set
        q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
            validation_loader=validation_loader, teacher=teacher,
            student=student, autoencoder=autoencoder,
            teacher_mean=teacher_mean, teacher_std=teacher_std,
            desc=f'Validation map normalization (epoch {epoch})'
        )

        test_output_dir = os.path.join(config.output_dir, "trainings", config.test_image_dir)
        auc = test(
            test_set=test_set, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start,
            q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
            test_output_dir=test_output_dir, 
            train_output_dir=train_output_dir,
            desc=f'Intermediate inference (epoch {epoch})'
        )
        
        # Log learning rate
        lr_student = optimizer_student.param_groups[0]['lr']
        lr_ae = optimizer_ae.param_groups[0]['lr']

        # Track student output distribution for debugging
        # with torch.no_grad():
        #     sample_img, _ = next(iter(train_loader))
        #     if on_gpu:
        #         sample_img = sample_img.cuda()
        #     student_out = student(sample_img)
        #     student_mean = student_out.mean().item()
        #     student_std = student_out.std().item()

        print(f'{GREEN}Epoch {epoch}: '
            f'test_auc={auc:.4f}, '
            f'st_loss={st_loss:.4f}, ae_loss={ae_loss:.4f}, stae_loss={stae_loss:.4f}, '
            f'lr_student={lr_student:.6f}, lr_ae={lr_ae:.6f}, '
            # f'student_out_mean={student_mean:.4f}, student_out_std={student_std:.4f}'
            f'{RESET}')
        
        # Save checkpoints (entire models + normalization parameters)
        save_path_student = os.path.join(train_output_dir, f'student_epoch{epoch}_auc{auc:.4f}.pth')
        save_path_autoencoder = os.path.join(train_output_dir, f'autoencoder_epoch{epoch}_auc{auc:.4f}.pth')
        
        # torch.save(student.state_dict(), save_path_student)
        # torch.save(autoencoder.state_dict(), save_path_autoencoder)
        
        torch.save(student, save_path_student)
        torch.save(autoencoder, save_path_autoencoder)
        torch.save({
                    'teacher_mean': teacher_mean,
                    'teacher_std': teacher_std,
                    'q_st_start': q_st_start,
                    'q_st_end': q_st_end,
                    'q_ae_start': q_ae_start,
                    'q_ae_end': q_ae_end
                }, os.path.join(train_output_dir, f'epoch{epoch}_normalization.pth'))

def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, 
         test_output_dir=None,
         train_output_dir=None,
         desc='Running inference'):
    y_true = []
    y_score = []
    st_score = []
    ae_score = []
    for image, target, path in tqdm(test_set, desc=desc):
        orig_width = image.width
        orig_height = image.height
        image = default_transform(image)
        image = image[None]
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')   # Bilinear interpolation resizes anomaly map to original image size
        map_combined = map_combined[0, 0].cpu().numpy()

        defect_class = os.path.basename(os.path.dirname(path))
        if test_output_dir is not None:
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
            tifffile.imwrite(file, map_combined)

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = image_level_score(map_combined, mode="max")  # Original code used max pooling for score; here we provide alternatives
        y_true.append(y_true_image)
        y_score.append(y_score_image)
        
        # Debug: separate scores for Student and AE branches
        m_st   = map_st[0,0].cpu().numpy()
        m_ae   = map_ae[0,0].cpu().numpy()
        score_st  = image_level_score(m_st,   mode='max')
        score_ae  = image_level_score(m_ae,   mode='max')
        st_score.append(score_st)
        ae_score.append(score_ae)
        
    auc_st = roc_auc_score(y_true=y_true, y_score=st_score) * 100
    auc_ae = roc_auc_score(y_true=y_true, y_score=ae_score) * 100
    auc = roc_auc_score(y_true=y_true, y_score=y_score) * 100
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    
    # Save PR curve
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f'PR curve (AP = {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    # plt.show()
    save_path_image = os.path.join(train_output_dir, f'pr_curve.png')
    plt.savefig(save_path_image, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"{GREEN}[AUC]{RESET} combined={GREEN}{auc:.2f}{RESET} | "
      f"st={YELLOW}{auc_st:.2f}{RESET} | "
      f"ae={BLUE}{auc_ae:.2f}{RESET} | "
      f"ap={ap:.4f}" )
    
    return auc

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    """
    Run forward passes of teacher, student, and autoencoder to generate anomaly maps.
    - image: input image tensor
    - teacher_mean, teacher_std: used to normalize teacher features
    - q_st_start/q_st_end, q_ae_start/q_ae_end: optional normalization ranges for score scaling
    Returns:
        map_combined: combined anomaly map (student-teacher + student-autoencoder)
        map_st: anomaly map from student-teacher difference
        map_ae: anomaly map from student-autoencoder difference
    """
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    
    # Student vs Teacher error map
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    # Student vs Autoencoder error map
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    # Optional quantile-based normalization
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    
    # Combine maps (equal weighting by default)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    
    return map_combined, map_st, map_ae

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    
    
    # Quantile thresholds: 90% as "normal" baseline, 99.5% as "extreme anomaly" bound
    q_st_start = torch.quantile(maps_st, q=0.9) # Student vs Teacher
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9) # Student vs Autoencoder
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader):

    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std



def test_only():
    config = get_argparse()
    config.mvtec_ad_path = "data"
    config.subdataset = 'screw'
    config.output_dir = "EfficientAD/output/1"
    
    dataset_path = config.mvtec_ad_path
    test_output_dir = os.path.join(config.output_dir, 'anomaly_maps',
                                   config.dataset, config.subdataset, 'test')
    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, config.subdataset, 'test'))

    # 同样加载 train_set 来做 teacher_normalization
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, config.subdataset, 'train'),
        transform=transforms.Lambda(train_transform))
    train_loader = DataLoader(full_train_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # 创建模型并加载权重
    model_folder = r'C:\Users\cui8szh\Documents\Temp\cv_anomaly_detection_tech\EfficientAD\output\1\trainings\mvtec_ad\screw'
    teacher = torch.load(os.path.join(model_folder, "teacher_tmp.pth"), map_location='cpu', weights_only=False)
    autoencoder = torch.load(os.path.join(model_folder, "autoencoder_tmp.pth"), map_location='cpu', weights_only=False)
    student = torch.load(os.path.join(model_folder, "student_tmp.pth"), map_location='cpu', weights_only=False)

    # 重新计算 mean/std
    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)
    
    # 如果需要，也可以跑 map_normalization
    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader=train_loader, teacher=teacher, student=student, autoencoder=autoencoder,
        teacher_mean=teacher_mean, teacher_std=teacher_std)

    # 测试
    auc = test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=test_output_dir, desc='Final inference')
    print('Final image auc: {:.4f}'.format(auc))
    

if __name__ == '__main__':
    main()
    # test_only()
