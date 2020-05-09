# Libraries
import torch
import time
import os
import argparse
from torch.utils.data import DataLoader
from datetime import datetime

# Local Files
from model import HourglassNet
from loss import L1, L1_alternate
from data import CelebData
from debug import debug


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training the network")
    parser.add_argument(
        '--epochs',
        default=10,
        type=int,
        help='the number of epochs to run',
    )
    parser.add_argument(
        '--batch',
        default=100,
        type=int,
        help='the batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='learning rate for the model'
    )
    parser.add_argument(
        '--data',
        default=30000,
        type=int,
        help='size of data to use'
    )
    parser.add_argument(
        '--model',
        default=None,
        help='name of the model to be saved'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='print additional information if true')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='debug model by outputting intermediate images')

    return parser.parse_args()

ARGS = parse_args()
# Settings
VERBOSE = bool(ARGS.verbose)
DEBUG = bool(ARGS.debug)

# Hyper parameters
EPOCHS = ARGS.epochs
BATCH_SIZE = ARGS.batch
LEARNING_RATE = ARGS.lr
MAX_DATA = ARGS.data

def train(model, optimizer, dataloader, epoch):

    num_batches = MAX_DATA // BATCH_SIZE

    epoch_loss = torch.tensor([0], dtype=torch.float32).cuda()

    for j, data in enumerate(dataloader, 0):
        I_sbatch, I_tbatch, L_sbatch, L_tbatch = data
        if epoch < 5:
            skip_count = 0
        elif epoch < 8:
            skip_count = epoch - 4
        else:
            skip_count = 4

        I_sbatch = torch.squeeze(I_sbatch, dim=1).cuda()
        L_tbatch = torch.squeeze(L_tbatch, dim=1).cuda()

        I_tbatch = torch.squeeze(I_tbatch, dim=1).cuda()
        L_sbatch = torch.squeeze(L_sbatch, dim=1).cuda()

        I_tp_batch, L_sp_batch = model.forward(I_sbatch, L_tbatch, skip_count)

        loss = L1(I_tbatch, I_tp_batch, L_sbatch, L_sp_batch)

        if (VERBOSE):
            print("Batch # {} / {} loss: {}".format(j + 1, num_batches, loss))

        epoch_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = epoch_loss / num_batches
    print("Epoch loss: ", epoch_loss)


model = HourglassNet(gray=True)
model.cuda()
model.train(True)
modelId = None
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

dataset = CelebData('../data/', int(ARGS.data))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
for i in range(EPOCHS):
    if (DEBUG):
        print("Outputing debug image.")
        if (i == 0):
            modelId = debug(model, i)
        else:
            debug(model, i, modelId)
        print("Finished outputting debug image. Continuing training")

    start = time.time()
    print("Training epoch #", i + 1, "/", EPOCHS)

    train(model, optimizer, dataloader, i)
    end = time.time()
    print("Time elapsed to train epoch #", i + 1, ":", end - start)

if ARGS.model is None:
    now = datetime.now()
    model_name = 'model_{}.pt'.format(now.strftime("%m-%d-%H%M"))
else:
    model_name = ARGS.model

print("Done training! Saving model as {}".format(model_name))
torch.save(model.state_dict(), os.path.join('../trained_models/', model_name))