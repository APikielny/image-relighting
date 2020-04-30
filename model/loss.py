import torch
from kornia.filters import SpatialGradient

def L1(I_t, I_tp, L_s, L_sp):
    img_l1 = torch.sum(torch.abs(I_t - I_tp))

    I_t_grad = SpatialGradient()(I_t)
    I_tp_grad = SpatialGradient()(I_tp)

    grad_l1 = torch.sum(torch.abs(I_t_grad - I_tp_grad))

    light_l2 = torch.sum((L_s - L_sp) ** 2)

    loss = ((img_l1 + grad_l1) / (128 * 128)) + (light_l2 / 9)
    return loss

def L1_alternate(I_t, I_tp, L_s, L_sp):
    num_images = I_t.size()[0]
    img_norm = torch.norm((I_t - I_tp), p=1, dim=3)  # computs l1 norm accross the columns
    img_norm = torch.max(img_norm) / num_images

    I_t_grad = SpatialGradient()(I_t)
    I_tp_grad = SpatialGradient()(I_tp)

    grad_norm = torch.norm((I_t_grad - I_tp_grad), p=1, dim=4)  # computs l1 norm accross the columns
    grad_norm = torch.max(grad_norm) / num_images

    image_loss = img_norm + grad_norm
    light_loss = torch.mean((L_s - L_sp) ** 2)

    loss = image_loss + light_loss

    return loss