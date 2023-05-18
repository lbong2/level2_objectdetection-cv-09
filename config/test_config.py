class Config:
    name = "resnet50_rpn_ciou_mosaic"
    model_weights = f"/opt/ml/level2_objectdetection-cv-09/save/Faster_R-CNN/{name}"
    image_path = "/opt/ml/dataset/test"
    gpu_id = '0'
    num_classes = 10 + 1
    data_root_dir = "/opt/ml/dataset"


test_cfg = Config()
