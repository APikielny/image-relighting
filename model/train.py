import torch
from model import HourglassNet
from loss import L1 #L1_batch, L1
from torch.utils.data import DataLoader
from data import CelebData
import time
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training the network")
    parser.add_argument(
        '--epochs',
        default=5,
        help='the number of EPOCHS to run',
    )
    parser.add_argument(
        '--batch',
        default='100',
        help='the batch size'
    )
    parser.add_argument(
        '--data',
        default='30000',
        help='size of data to use'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='print additional information if true')

    return parser.parse_args()


ARGS = parse_args()
EPOCHS = int(ARGS.epochs)
BATCH_SIZE = int(ARGS.batch)
MAX_DATA = int(ARGS.data)
VERBOSE = bool(ARGS.verbose)

# 5 epochs, 0 skip
# add one for subsequent for next 4

def train(model, optimizer, dataloader, epoch):

    num_batches = MAX_DATA // BATCH_SIZE

    epoch_loss = torch.tensor([0], dtype=torch.float32).cuda()

    for j, data in enumerate(dataloader, 0):
        #total_loss = torch.tensor([0], dtype=torch.float32).cuda()
        I_sbatch, I_tbatch, L_sbatch, L_tbatch = data

        # for k in range(BATCH_SIZE):
        #     I_s = I_sbatch[k]
        #     I_t = I_tbatch[k]
        #     L_s = L_sbatch[k]
        #     L_t = L_tbatch[k]

        #     skip_count = 4
        #     I_tp, L_sp = model.forward(I_s, L_t, skip_count)

        #     N = I_s.shape[0] * I_s.shape[0]
        #     loss = L1(N, I_t, I_tp, L_s, L_sp)
        #     total_loss += loss

        if epoch < 5:
            skip_count = 0
        elif epoch == 5:
            skip_count = 1
        elif epoch == 6:
            skip_count = 2
        elif epoch == 7:
            skip_count = 3
        else:
            skip_count = 4
        # print("I_s batch shape:", I_sbatch.shape)
        # print("L_t batch shape:", L_tbatch.shape)

        I_sbatch = torch.squeeze(I_sbatch, dim=1).cuda()
        L_tbatch = torch.squeeze(L_tbatch, dim=1).cuda()

        I_tbatch = torch.squeeze(I_tbatch, dim=1).cuda()
        L_sbatch = torch.squeeze(L_sbatch, dim=1).cuda()

        # print("I_s batch shape squeeze:", I_sbatch.shape)
        # print("L_t batch shape squeeze:", L_tbatch.shape)

        I_tp_batch, L_sp_batch = model.forward(I_sbatch, L_tbatch, skip_count)

        N = I_sbatch.shape[2] * I_sbatch.shape[2]
        loss = L1(N, I_tbatch, I_tp_batch, L_sbatch, L_sp_batch)
        # total_loss += loss

        #total_loss = total_loss / BATCH_SIZE
        if (VERBOSE):
            print("Batch # {} / {} loss: {}".format(j, num_batches, loss))

        epoch_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = epoch_loss / num_batches
    print("Epoch loss: ", epoch_loss)


model = HourglassNet(gray=True)
model.cuda()
model.train(True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

dataset = CelebData('../data/', int(ARGS.data))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
for i in range(EPOCHS):
    start = time.time()
    print("Training epoch #", i + 1, "/", EPOCHS)

    train(model, optimizer, dataloader, i)
    end = time.time()
    print("Time elapsed to train epoch #", i + 1, ":", end - start)

print("Done training! Saving model.")
num_models = len(os.listdir('../trained_models/'))
torch.save(model.state_dict(), '../trained_models/model_{:d}.pt'.format(num_models + 1))