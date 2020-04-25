import torch
from model import HourglassNet
from loss import L1
from random import shuffle
from data import load_data

EPOCHS = 10
BATCH_SIZE = 100

def train(model, optimizer, data):

    num_batches = len(data) // BATCH_SIZE

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
        print(total_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

model = HourglassNet(gray=False)
model.train(True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
data = load_data('data/train/')

for i in range(EPOCHS):
    shuffle(data)
    train(model, optimizer, data)

# TODO: Save model
