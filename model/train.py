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


def train(model, optimizer, data):

    num_batches = len(data) // BATCH_SIZE
    # if (VERBOSE):
    #     print("Num batches: ", num_batches)

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


        skip_count = 4
        # print("I_s batch shape:", I_sbatch.shape)
        # print("L_t batch shape:", L_tbatch.shape)

        I_sbatch = torch.squeeze(I_sbatch, dim=1)
        L_tbatch = torch.squeeze(L_tbatch, dim=1)

        I_tbatch = torch.squeeze(I_sbatch, dim=1)
        L_sbatch = torch.squeeze(L_tbatch, dim=1)

        print("I_s batch shape squeeze:", I_sbatch.shape)
        print("L_t batch shape squeeze:", L_tbatch.shape)

        I_tp_batch, L_sp_batch = model.forward(I_sbatch, L_tbatch, skip_count)

        N = I_sbatch.shape[2] * I_sbatch.shape[2]
        loss = L1(N, I_tbatch, I_tp_batch, L_sbatch, L_sp_batch)
        # total_loss += loss

        #total_loss = total_loss / BATCH_SIZE
        if (VERBOSE):
            print("Batch loss:", loss)

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
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
for i in range(EPOCHS):
    start = time.time()
    print("Training epoch #", i + 1, "/", EPOCHS)

    train(model, optimizer, dataloader)
    end = time.time()
    print("Time elapsed to train epoch #", i + 1, ":", end - start)

print("Done training! Saving model.")
num_models = len(os.listdir('../trained_models/'))
torch.save(model.state_dict(), '../trained_models/model_{:d}.pt'.format(num_models + 1))