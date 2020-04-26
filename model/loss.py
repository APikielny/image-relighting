from skimage import filters
import torch

# def L1(N, I_t, I_tp, L_s, L_sp):

#     img_norm = torch.mean(torch.abs(I_t - I_tp))

#     #grad_norm = torch.norm(I_t.grad - I_tp.grad)

#     grad_norm = 0
#     if (I_t.grad != None):
#         print("I_t not none!")
#         grad_norm = torch.mean(torch.abs(I_t.grad - I_tp.grad))


#     # np_grad = filters.gaussian(I_t.cpu().detach().numpy()) - filters.gaussian(I_tp.cpu().detach().numpy())
#     # grad_norm = torch.norm(torch.from_numpy(np_grad))

    
#     image_loss = img_norm + grad_norm

#     light_loss = torch.sum((L_s - L_sp) ** 2)


#     loss = image_loss + light_loss

    
#     return loss

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
