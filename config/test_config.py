class Config:
    model_weights = "/opt/ml/level2_objectdetection-cv-09/test/mobilenet-model-0-mAp-0.022099911296862215.pth"
    image_path = "/opt/ml/dataset/test"
    gpu_id = '0'
    num_classes = 10 + 1
    data_root_dir = "/opt/ml/dataset"


test_cfg = Config()
