from torch.optim import SGD, Adagrad, Adam


OPTIMIZER_ENTRYPOINTS = {
    'sgd': SGD,
    'adagrad': Adagrad,
    'adam': Adam,
}

def optimizer_entrypoint(optimizer_name):
    """_summary_
    OPTIMIZER_ENTRYPOINTS에 해당하는 optimizer return

    Args:
        optimizer_name (str): optimizer name

    Returns:
        optimizer (class): optimizer
    """
    return OPTIMIZER_ENTRYPOINTS[optimizer_name]


def is_optimizer(optimizer_name):
    """_summary_
    OPTIMIZER_ENTRYPOINTS에 해당하는 optimizer인지 확인

    Args:
        optimizer_name (str): optimizer name

    Returns:
        bool: 있다면 True, 없으면 False
    """
    return optimizer_name in OPTIMIZER_ENTRYPOINTS


def create_optimizer(optimizer_name, parameter, args):
    """_summary_

    Args:
        optimizer_name (str): ['sgd', 'adagrad', 'adam'] 사용가능

    Raises:
        RuntimeError: 해당 하는 optimizer가 없다면 raise error

    Returns:
        optimizer (Module): 해당 하는 optimizer return
    """
    if is_optimizer(optimizer_name):
        create_fn = optimizer_entrypoint(optimizer_name)
        if optimizer_name in 'sgd':
            optimizer = create_fn(
                parameter,
                lr= args.lr,
                momentum = args.momentum,
                weight_decay = args.weight_decay
                )

        else:
            optimizer = create_fn(
                parameter,
                lr= args.lr,
                weight_decay = args.weight_decay
                )
    else:
        raise RuntimeError('Unknown optimizer (%s)' % optimizer_name)
    return optimizer