from skimage import filters
import torch

def L1(N, I_t, I_tp, L_s, L_sp):

    img_norm = torch.norm(I_t - I_tp)
    if I_t.grad is None or I_tp.grad is None:
        grad_norm = 0
    else:
        grad_norm = torch.norm(I_t.grad - I_tp.grad)
    image_loss = img_norm + grad_norm
    light_loss = (L_s - L_sp) ** 2

    loss = (1/N) * image_loss + light_loss
    return loss
