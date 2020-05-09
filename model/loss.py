import torch
from kornia.filters import SpatialGradient

# Loss function for training

def L1(I_t, I_tp, L_s, L_sp):
    img_l1 = torch.sum(torch.abs(I_t - I_tp))

    I_t_grad = SpatialGradient()(I_t)
    I_tp_grad = SpatialGradient()(I_tp)

    grad_l1 = torch.sum(torch.abs(I_t_grad - I_tp_grad))

    light_l2 = torch.sum((L_s - L_sp) ** 2)

    N = I_t.shape[2]
    loss = ((img_l1 + grad_l1) / (N * N)) + (light_l2 / 9)
    return loss