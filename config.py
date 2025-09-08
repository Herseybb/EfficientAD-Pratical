import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subdataset', default='bottle',
                        help='One of 15 sub-datasets of Mvtec AD or 5' +
                             'sub-datasets of Mvtec LOCO')
    parser.add_argument('-o', '--output_dir', default='output/1')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='models/teacher_small.pth')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='./mvtec_anomaly_detection',
                        help='Downloaded Mvtec AD dataset')
    parser.add_argument('-e', '--epochs', type=int, default=100)
    return parser.parse_args()


# def get_config():
#     config = get_argparse()
    
#     config.dataset_path = ""
#     config.subdataset = None
#     config.train_path = 'train'
#     config.test_path = 'test'
    
#     config.output_dir = "output"
#     config.test_image_dir = "images"
#     config.weights = "models/teacher_final.pth"
#     config.student_weights = None
#     config.ae_weights = None
#     config.epochs = 50
    
#     return config

def get_config():
    config = get_argparse()
    
    config.dataset_path = r"C:\Users\cui8szh\Documents\Temp\cv_anomaly_detection_tech\data\pincheck"
    config.subdataset = None
    config.train_path = 'Pass'
    config.test_path = 'Fail'
    
    config.output_dir = "EfficientAD-Pratical/output/pincheck"
    config.weights = r"C:\Users\cui8szh\Documents\Temp\cv_anomaly_detection_tech\EfficientAD\models\teacher_final.pth"
    config.student_weights = None
    config.ae_weights = None
    config.epochs = 50
    
    return config
    