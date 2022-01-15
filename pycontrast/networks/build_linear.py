import torch.nn as nn


def build_linear(opt):
    n_class = opt.n_class
    arch = opt.arch
    if arch.endswith('x4'):
        n_feat = 2048 * 4
    elif arch.endswith('x2'):
        n_feat = 2048 * 2
    else:
        n_feat = 2048
    classifier = nn.Sequential(nn.Linear(n_feat,int(n_feat//2**2)),nn.ReLU(inplace=True),
                 nn.Linear(int(n_feat//2**2),int(n_feat//2**4)),nn.ReLU(inplace=True),
                 nn.Linear(int(n_feat//2**4),int(n_feat//2**6)),nn.ReLU(inplace=True),
                 nn.Linear(int(n_feat//2**6),n_class),)
    # classifier = nn.Linear(n_feat, n_class)
    return classifier
