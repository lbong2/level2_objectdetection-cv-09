from torch.optim.lr_scheduler import StepLR, LambdaLR, ExponentialLR, CosineAnnealingLR, CyclicLR, ReduceLROnPlateau


SCHEDULER_ENTRYPOINTS = {
    'steplr': StepLR,
    'lambdalr': LambdaLR,
    'exponentiallr': ExponentialLR,
    'cosineannealinglr': CosineAnnealingLR,
    'cycliclr': CyclicLR,
    'reducelronplateau': ReduceLROnPlateau
}


def scheduler_entrypoint(scheduler_name):
    """_summary_
    SCHEDULER_ENTRYPOINTS에 해당하는 scheduler return

    Args:
        scheduler_name (str): scheduler name

    Returns:
        scheduler (class): scheduler
    """
    return SCHEDULER_ENTRYPOINTS[scheduler_name]


def is_scheduler(scheduler_name):
    """_summary_
    SCHEDULER_ENTRYPOINTS에 해당하는 scheduler인지 확인

    Args:
        scheduler_name (str): scheduler name

    Returns:
        bool: 있다면 True, 없으면 False
    """
    return scheduler_name in SCHEDULER_ENTRYPOINTS


def create_scheduler(scheduler_name, optimizer, args):
    """_summary_

    Args:
        scheduler_name (str): ['steplr', 'lambdalr', 'exponentiallr', 'cosineannealinglr', 'cycliclr', 'reducelronplateau'] 사용가능

    Raises:
        RuntimeError: 해당 하는 scheduler가 없다면 raise error

    Returns:
        scheduler (Module): 해당 하는 scheduler return
    """
    if is_scheduler(scheduler_name):
        create_fn = scheduler_entrypoint(scheduler_name)
        kargs = {}
        if scheduler_name =="steplr":
            kargs['step_size'] = args.lr_decay_step
            kargs['gamma'] = args.lr_gamma
        
        elif scheduler_name =="lambdalr":
            kargs['lr_lambda'] = args.lr_lambda
        
        elif scheduler_name =="exponentialLR":
            kargs['gamma'] = args.lr_gamma
        
        elif scheduler_name == "cosineannealinglr":
            kargs['T_max'] = args.tmax
            kargs['eta_min'] = args.lr*0.01
        
        elif scheduler_name == "cycliclr":
            kargs['base_lr'] = args.lr
            kargs['max_lr'] = args.maxlr
            kargs['step_size_up'] = args.tmax
            kargs['mode'] = 'triangular'
        
        elif scheduler_name == "reducelronplateau":
            kargs['mode'] = "max"
            kargs['factor'] = args.lr_gamma
            kargs['patience'] = args.patience
            kargs['threshold'] = args.threshold

        scheduler = create_fn(optimizer,**kargs)
    else:
        raise RuntimeError('Unknown scheduler (%s)' % scheduler_name)
    return scheduler