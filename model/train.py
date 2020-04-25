import torch
from model import HourglassNet
from loss import L1
from data import load_data
from random import shuffle
import time

EPOCHS = 5
BATCH_SIZE = 100

def train(model, optimizer, data):

    num_batches = len(data) // BATCH_SIZE
    epoch_loss = 0
    
    for i in range(num_batches):
        total_loss = 0
        for j in range(i * BATCH_SIZE, min(i * BATCH_SIZE + BATCH_SIZE, len(data))):
            I_s = data[j].I_s
            I_t = data[j].I_t
            L_s = data[j].L_s
            L_t = data[j].L_t

            skip_count = 4
            I_tp, L_sp = model.forward(I_s, L_t, skip_count)

            N = I_s.shape[0] * I_s.shape[0]
            loss = L1(N, I_t, I_tp, L_s, L_sp)
            total_loss += loss

        total_loss = torch.mean(total_loss)
        epoch_loss += total_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    print("Epoch loss: ", epoch_loss)


model = HourglassNet(gray=True)
model.cuda()
model.train(True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print("Loading data.")
start = time.time()
data = load_data('../data/')
end = time.time()
print("Loaded data. Size: ", len(data))
print("Time elapsed:", end - start)

for i in range(EPOCHS):
    start = time.time()
    print("Training epoch #", i + 1, "/", EPOCHS)
    shuffle(data)
    train(model, optimizer, data)
    end = time.time()
    print("Time elapsed to train epoch #", i + 1,":", end - start)

print("Done training! Saving model.")
num_models = len(os.listdir('../trained_models/'))
torch.save(model.state_dict(), '../trained_models/model_{:d}.pt'.format(num_models +  1))
