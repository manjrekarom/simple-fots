import torch


def move_to(device, *args):
    new_args = []
    for arg in args:
        if torch.is_tensor(arg):
            new_args.append(arg.to(device))
        else:
            new_args.append(arg)
    return new_args
