

default_config ={
    'backbone' : 'resnet50_fpn',  # [mobilenet, resnet50_fpn]
    'backbone_pretrained_weights' : "/opt/ml/level2_objectdetection-cv-09/pretrained/fasterrcnn_resnet50_fpn.pth",  # [path or None]
    # /opt/ml/level2_objectdetection-cv-09/pretrained/mobilenet_v2.pth
    # /opt/ml/level2_objectdetection-cv-09/pretrained/fasterrcnn_resnet50_fpn.pth
    
    # data transform parameter
    'train_horizon_flip_prob' : 0.5,  # data horizon flip probility in train transform
    'min_size' : 800,
    'max_size' : 1333,
    'image_mean' : [0.485, 0.456, 0.406],
    'image_std' : [0.229, 0.224, 0.225],

    # anchor parameters
    'anchor_size' : [32, 64, 128, 256, 512], # if resnet50, [32, 64, 128, 256, 512] not [64,128,256]
    'anchor_ratio' : [0.5, 1, 2.0],

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
    'box_score_thresh' : 0.01,
    'box_nms_thresh' : 0.5,
    'box_detections_per_img' : 100,
    'box_fg_iou_thresh' : 0.5,
    'box_bg_iou_thresh' : 0.5,
    'box_batch_size_per_image' : 512,
    'box_positive_fraction' : 0.25,
    'bbox_reg_weights' : None,

    'device' : 'cuda:0',
    'seed' : 42,

    'resume' : '',  # pretrained_weights
    'start_epoch' : 0,  # start epoch
    'num_epochs' : 100,  # train epochs
    'early_stop' : 10, # early_stop

    # learning rate parameters
    'optimizer' : 'sgd', # ['sgd','adagrad','adam']
    'lr' : 1e-3,
    'momentum' : 0.9,
    'weight_decay' : 0.0005,
    
    # learning rate schedule
    'scheduler' : 'cosineannealinglr', # ['steplr', 'lambdalr', 'exponentiallr', 'cosineannealinglr','cycliclr','reducelronplateau']
    'lr_gamma' : 0.33, # using steplr, exponentialLR, reducelronplateau
    'lr_decay_step' : 20, # using steplr
    'tmax' : 5, # using cosineannealinglr
    'maxlr' : 0.01, # using cycliclr
    'patience' : 2, # using reducelronplateau
    'threshold' : 1e-4, # using reducelronplateau

    'batch_size' : 16,
    'mosaic' : True,

    'num_class' : 10 + 1,  # foreground + 1 background
    'num_workers' : 8,

    'box_loss': 'smoothl1loss', # ['smoothl1loss', 'iou_loss', 'giou_loss', 'diou_loss', 'ciou_loss']
    'cls_loss':'label_smoothing', # ['focal', 'cross_entropy', 'label_smoothing']
    'rpn_box_loss': 'smoothl1loss', # ['smoothl1loss', 'iou_loss', 'giou_loss', 'diou_loss', 'ciou_loss']
    'rpn_cls_loss': 'bce', # ['bce']
    'loss_gain':[1, 1, 1, 1], # ['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']

    # wandb project
    'wandb' : True,
    'project': "Faster_R-CNN",
    'name': "resnet50_rpn_diou_mosaic_labelsmooth",
    'notes' : "mosaic test",
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

train_cfg = ChangeConfig(default_config)