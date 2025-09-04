import torch
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
from torchvision import transforms

# constants
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

# data loading
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model_and_stats(model_dir):
    """
    加载 teacher、student、autoencoder 模型以及训练时保存的归一化统计量。

    Args:
        model_dir (str): 模型和统计量文件所在文件夹路径。

    Returns:
        dict: 包含以下键值：
            - teacher
            - student
            - autoencoder
            - teacher_mean
            - teacher_std
            - q_st_start
            - q_st_end
            - q_ae_start
            - q_ae_end
    """
    # 构建文件路径
    teacher_path = os.path.join(model_dir, 'teacher_final.pth')
    student_path = os.path.join(model_dir, 'student_final.pth')
    autoencoder_path = os.path.join(model_dir, 'autoencoder_final.pth')
    stats_path = os.path.join(model_dir, 'normalization.pth')

    # 检查文件是否存在
    for f in [teacher_path, student_path, autoencoder_path, stats_path]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"文件不存在: {f}")

    # 加载模型
    teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)
    student = torch.load(student_path, map_location='cpu', weights_only=False)
    autoencoder = torch.load(autoencoder_path, map_location='cpu', weights_only=False)

    # 加载统计量
    stats = torch.load(stats_path)
    teacher_mean = stats['teacher_mean']
    teacher_std = stats['teacher_std']
    q_st_start = stats['q_st_start']
    q_st_end = stats['q_st_end']
    q_ae_start = stats['q_ae_start']
    q_ae_end = stats['q_ae_end']

    return {
        'teacher': teacher,
        'student': student,
        'autoencoder': autoencoder,
        'teacher_mean': teacher_mean,
        'teacher_std': teacher_std,
        'q_st_start': q_st_start,
        'q_st_end': q_st_end,
        'q_ae_start': q_ae_start,
        'q_ae_end': q_ae_end
    }
    
@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae


def infer_single_image(image_path, teacher, student, autoencoder,
                       teacher_mean, teacher_std, q_st_start, q_st_end, q_ae_start, q_ae_end,
                       threshold=0.2, resize=256):
    # 1. 读取图片
    image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = image.size
    # 2. transform
    image_tensor = default_transform(image)[None]

    # 3. predict
    map_combined, map_st, map_ae = predict(
        image=image_tensor, teacher=teacher, student=student, autoencoder=autoencoder,
        teacher_mean=teacher_mean, teacher_std=teacher_std,
        q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end
    )
    map_combined = torch.nn.functional.interpolate(
        map_combined, (orig_height, orig_width), mode='bilinear')[0,0].cpu().numpy()
    
    # 4. anomaly score
    score = np.max(map_combined)   # todo 这个score怎么设置有待商榷
    status = "Fail" if score > threshold else "Pass"
    return score, status, map_combined


def test_single_img(img_path):
    model_dir = r"C:\Users\cui8szh\Documents\Temp\cv_anomaly_detection_tech\EfficientAD\output\1\trainings\mvtec_ad\screw"
    models_stats = load_model_and_stats(model_dir)

    teacher = models_stats['teacher']
    student = models_stats['student']
    autoencoder = models_stats['autoencoder']
    teacher_mean = models_stats['teacher_mean']
    teacher_std = models_stats['teacher_std']
    q_st_start = models_stats['q_st_start']
    q_st_end = models_stats['q_st_end']
    q_ae_start = models_stats['q_ae_start']
    q_ae_end = models_stats['q_ae_end']
    
    score, status, map_combined = infer_single_image(
            img_path, teacher, student, autoencoder,
            teacher_mean, teacher_std, q_st_start, q_st_end, q_ae_start, q_ae_end,
            threshold=0.2
        )
    print(f"Score: {score:.4f}, Status: {status}")
    
    # show map
    # import matplotlib.pyplot as plt

    # # Show array as image
    # plt.imshow(map_combined, cmap="gray")   # remove cmap="gray" if it's RGB
    # plt.axis("off")
    # plt.show()
    return score, status, map_combined


def save_image_safe(arr: np.ndarray, path: str):
    arr = np.array(arr, dtype=float)  # 确保是浮点
    arr_min, arr_max = arr.min(), arr.max()

    if arr_max == arr_min:
        # 如果整张图都是一个值，就直接设为 0
        norm = np.zeros_like(arr, dtype=np.uint8)
    else:
        # 归一化到 0–255
        norm = (255 * (arr - arr_min) / (arr_max - arr_min)).astype(np.uint8)

    # 保存
    Image.fromarray(norm).save(path)
    
def temp_test_mvtech_all():
    input_dir = r"C:\Users\cui8szh\Documents\Temp\cv_anomaly_detection_tech\data\screw\test"
    output_dir = r"C:\Users\cui8szh\Documents\Temp\cv_anomaly_detection_tech\EfficientAD\output\1\test"
    
    exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}  # 常见图片后缀
    label = ""
    results = []
    for folder in tqdm(os.listdir(input_dir), desc="文件夹进度"):
        label = "Pass" if folder == "good" else "Fail"
        path = os.path.join(input_dir, folder)
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            _, ext = os.path.splitext(file_path.lower())
            if ext in exts:
                score, status, map_combined = test_single_img(file_path)
                results.append({
                    "label": label,
                    "folder": folder,
                    "filename": file_name,
                    "score": score,
                    "status": status
                })
                
                img_out_dir = os.path.join(output_dir, folder)
                os.makedirs(img_out_dir, exist_ok=True)
                out_name = file_name.split(".")[0] + f"_{score:.2f}_{status}" + ".png"
                save_image_safe(map_combined, os.path.join(img_out_dir, out_name))
                
                
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, "results.csv"), index=None)
                

        

if __name__ == "__main__":
    
    # === single image ===
    # img_path = r"C:\Users\cui8szh\Documents\Temp\cv_anomaly_detection_tech\data\screw\test\good\000.png"
    # test_single_img(img_path)
    
    # === image folder ===
    temp_test_mvtech_all()