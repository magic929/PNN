import torch

def truncated_normal_(size, mean=0, std=0.09):
    tensor = torch.zeros(size)
    tmp = tensor.new_empty(size+(4, )).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

