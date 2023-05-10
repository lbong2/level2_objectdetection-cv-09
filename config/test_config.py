class Config:
    name = "backbone_(manipulated variable)"
    model_weights = f"save/{name}"
    image_path = "/opt/ml/dataset/test"
    gpu_id = '0'
    num_classes = 10 + 1
    data_root_dir = "/opt/ml/dataset"


test_cfg = Config()
