import os, argparse
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split

from tqdm import tqdm

import model

parser = argparse.ArgumentParser(description='Training and Using CNN')
parser.add_argument('--train-data', metavar='DIR', help='path to training dataset')
parser.add_argument('--evaluate-data', metavar='DIR', help='path to evaluation dataset')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N', help='size of mini-batch (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='learning rate at start of training')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='use this flag to validate without training')

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


def main():
    model = Network().to(device)
    criterion = nn.CrossEntropyLoss()
    print('Your network:')
    print(summary(model, (1,28,28), device=device))

    # Set up optimization hyperparameters

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight-decay)


    start_time = time.time()
    trn_loss_hist, trn_acc_hist, val_acc_hist = train(model, args.train-data, args.epochs)
    end_time = time.time()

    print(f"Total time to train the model: {(end_time - start_time):.3f}")


    print("\n Evaluate on test set")
    evaluate(model, args.evaluate-data)


if __name__ == '__main__':
    main()
