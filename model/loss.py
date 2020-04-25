from skimage import filters
import torch

def L1(N, I_t, I_tp, L_s, L_sp):

    img_norm = torch.norm(I_t - I_tp)
    np_grad = filters.gaussian(I_t.numpy()) - filters.gaussian(I_tp.numpy())
    grad_norm = torch.norm(torch.from_numpy(np_grad))
    image_loss = img_norm + grad_norm

    light_loss = (L_s - L_sp) ** 2

    loss = (1/N) * image_loss + light_loss
    return loss
