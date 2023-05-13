

default_config ={
    'backbone' : 'mobilenet',  # [vgg16, resnet-fpn, mobilenet, resnet50_fpn]
    'backbone_pretrained_weights' : "/opt/ml/level2_objectdetection-cv-09/pretrained/mobilenet_v2.pth",  # [path or None]

    # data transform parameter
    'train_horizon_flip_prob' : 0.5,  # data horizon flip probility in train transform
    'min_size' : 512,
    'max_size' : 1024,
    'image_mean' : [0.485, 0.456, 0.406],
    'image_std' : [0.229, 0.224, 0.225],

    # anchor parameters
    'anchor_size' : [64, 128, 256, 512],
    'anchor_ratio' : [0.25, 0.5, 1, 2.0, 4.0],

    # roi align parameters
    'roi_out_size' : [7, 7],
    'roi_sample_rate' : 2,

    # rpn process parameters
    'rpn_pre_nms_top_n_train' : 2000,
    'rpn_post_nms_top_n_train' : 2000,

    'rpn_pre_nms_top_n_test' : 1000,
    'rpn_post_nms_top_n_test' : 1000,

    'rpn_nms_thresh' : 0.7,
    'rpn_fg_iou_thresh' : 0.7,
    'rpn_bg_iou_thresh' : 0.3,
    'rpn_batch_size_per_image' : 256,
    'rpn_positive_fraction' : 0.5,

    # remove low threshold target
    'box_score_thresh' : 0.25,
    'box_nms_thresh' : 0.5,
    'box_detections_per_img' : 100,
    'box_fg_iou_thresh' : 0.5,
    'box_bg_iou_thresh' : 0.5,
    'box_batch_size_per_image' : 512,
    'box_positive_fraction' : 0.25,
    'bbox_reg_weights' : None,

    'device_name' : 'cuda:0',

    'resume' : '',  # pretrained_weights
    'start_epoch' : 0,  # start epoch
    'num_epochs' : 10,  # train epochs
    'early_stop' : 5, # early_stop

    # learning rate parameters
    'optimizer' : 'adam', # ['sgd','adagrad','adam']
    'lr' : 1e-4,
    'momentum' : 0.8,
    'weight_decay' : 0.0005,
    
    # learning rate schedule
    'scheduler' : 'reducelronplateau', # ['steplr', 'lambdalr', 'exponentiallr', 'cosineannealinglr','cycliclr','reducelronplateau']
    'lr_gamma' : 0.33, # using steplr, exponentialLR, reducelronplateau
    'lr_decay_step' : 20, # using steplr
    'tmax' : 5, # using cosineannealinglr
    'maxlr' : 0.01, # using cycliclr
    'patience' : 3, # using reducelronplateau
    'threshold' : 1e-4, # using reducelronplateau

    'batch_size' : 16,

    'num_class' : 10 + 1,  # foreground + 1 background

    'num_workers' : 0,

    # wandb project
    'wandb' : True,
    'project': "Faster_R-CNN",
    'name': "Horizon",
    'notes' : "albumentation test",
    'entity' : "boost_cv_09",

    'data_root_dir' : "/opt/ml/dataset",
    'model_save_dir' : "save/",
}

sweep_config ={
    'method' : 'grid',
    'project' : "sweep Faster_R-CNN",
    'name' : 'sweep_{}',
    'metric':{
        'name':'val_mAP',
        'goal':'maximize'
    },
    'parameters':
    {
        'optimizer': {
            'values':['adam','sgd']
        },
        'lr':{
            'values':[1e-3, 1e-4, 1e-5]
        },
        'batch_size':{
            'values':[4,16]
        }

    },
}

class ChangeConfig(object):
    def __init__(self,cfg_dict):
        self.__dict__.update(cfg_dict)


