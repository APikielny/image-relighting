from skimage import filters
import torch
from kornia.filters import SpatialGradient

def L1(N, I_t, I_tp, L_s, L_sp):

    # print("in single loss!")

    # print("I_t shape", I_t.shape)
    # print("I_tp shape", I_tp.shape)

    # print("abs shape", torch.abs(I_t - I_tp).shape)

    #img_norm = torch.mean(torch.abs(I_t - I_tp)) #this takes the mean across all dimensions which i think is fine?
    # print("img norm:", img_norm)
    # print("img norm shape:", img_norm.shape)
    #L1 norm of a matrix is the max of the l1 norms of the columns of the matrix
    

    # img_grad = torch.mean(torch.abs(I_t_grad - I_tp_grad)) #what is this?
  
    # grad_norm = 0
    # if (I_t.grad != None):
    #     print("I_t not none!")
    #     grad_norm = torch.mean(torch.abs(I_t.grad - I_tp.grad))

    # grad_norm = torch.mean(torch.abs(I_t_grad - I_tp_grad))
    
    # np_grad = filters.gaussian(I_t.cpu().detach().numpy()) - filters.gaussian(I_tp.cpu().detach().numpy())
    # grad_norm = torch.norm(torch.from_numpy(np_grad))

    num_images = I_t.size()[0]
    img_norm = torch.norm((I_t - I_tp), p=1, dim=3) #computs l1 norm accross the columns
    img_norm = torch.max(img_norm)/num_images

    I_t_grad = SpatialGradient()(I_t)
    I_tp_grad = SpatialGradient()(I_tp)
    
    grad_norm = torch.norm((I_t_grad - I_tp_grad), p=1, dim=4) #computs l1 norm accross the columns
    grad_norm = torch.max(grad_norm)/num_images

    image_loss = img_norm + grad_norm
    light_loss = torch.mean((L_s - L_sp) ** 2)

    # print("Ls shape", L_s.shape)
    # print("Lsp shape", L_sp.shape)

    # print("light loss:", light_loss)
    # print("light loss shape:", light_loss.shape)


    loss = image_loss + light_loss

    
    return loss

def L1_test(I_t, I_tp, L_s, L_sp):


    img_l1 = torch.sum(torch.abs(I_t - I_tp))

    I_t_grad = SpatialGradient()(I_t)
    I_tp_grad = SpatialGradient()(I_tp)

    grad_l1 = torch.sum(torch.abs(I_t_grad - I_tp_grad))

    light_l2 = torch.sum((L_s - L_sp) ** 2)

    #loss = 1/(100*128*128) * (img_l1 + grad_l1) + light_l2
    loss = 1/(100*128*128) * img_l1
    return loss