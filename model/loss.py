from skimage import filters
import torch
from kornia.filters import SpatialGradient

def L1(N, I_t, I_tp, L_s, L_sp):

    print("in single loss!")

    print("I_t shape", I_t.shape)
    print("I_tp shape", I_tp.shape)

    print("abs shape", torch.abs(I_t - I_tp).shape)

    img_norm = torch.mean(torch.abs(I_t - I_tp)) #this takes the mean across all dimensions which i think is fine?
    print("img norm:", img_norm)
    print("img norm shape:", img_norm.shape)

    I_t_grad = SpatialGradient()(I_t)
    I_tp_grad = SpatialGradient()(I_tp)

    img_grad = torch.mean(torch.abs(I_t_grad - I_tp_grad))

    # grad_norm = 0
    # if (I_t.grad != None):
    #     print("I_t not none!")
    #     grad_norm = torch.mean(torch.abs(I_t.grad - I_tp.grad))


    # np_grad = filters.gaussian(I_t.cpu().detach().numpy()) - filters.gaussian(I_tp.cpu().detach().numpy())
    # grad_norm = torch.norm(torch.from_numpy(np_grad))

    
    image_loss = img_norm + img_grad

    light_loss = torch.mean((L_s - L_sp) ** 2)

    print("Ls shape", L_s.shape)
    print("Lsp shape", L_sp.shape)

    print("light loss:", light_loss)
    print("light loss shape:", light_loss.shape)


    loss = image_loss + light_loss

    
    return loss

#L1 loss for a whole batch
def L1_batch(N, I_t, I_tp, L_s, L_sp):

    print("loss function.")

    print("all losses:", torch.mean(torch.abs(I_t - I_tp)))
    print("all losses shape:", torch.mean(torch.abs(I_t - I_tp)).shape)

    print("mean loss: ", torch.mean(torch.mean(torch.abs(I_t - I_tp))))

    img_norm = torch.mean(torch.mean(torch.abs(I_t - I_tp)))

    #grad_norm = torch.norm(I_t.grad - I_tp.grad)

    # grad_norm = 0
    # if (I_t.grad != None):
    #     print("I_t not none!")
    #     grad_norm = torch.mean(torch.abs(I_t.grad - I_tp.grad))


    # np_grad = filters.gaussian(I_t.cpu().detach().numpy()) - filters.gaussian(I_tp.cpu().detach().numpy())
    # grad_norm = torch.norm(torch.from_numpy(np_grad))

    
    image_loss = img_norm #+ grad_norm

    print("all light loss:", (L_s - L_sp) ** 2)
    light_loss = torch.sum((L_s - L_sp) ** 2)

    print("light loss:", (L_s - L_sp) ** 2)


    loss = image_loss + light_loss

    
    return loss
