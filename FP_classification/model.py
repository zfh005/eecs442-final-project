import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class Network(nn.Module):       # {Network} quoted from (lukemelas.github.io) for testing
    def __init__(self):
        super(Network, self).__init__()
        MIDLEVEL_FEATURE_SIZE = 128

        ## First half: ResNet
        resnet = models.resnet18(num_classes=365)
        # Change first conv layer to accept single-channel (grayscale) input
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1))
        # Extract midlevel features from ResNet-gray
        self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

        ## Second half: Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):       # {Forward} quoted from (lukemelas.github.io) for testing
        # Pass input through ResNet-gray to extract features
        midlevel_features = self.midlevel_resnet(input)

        # Upsample to get colors
        output = self.upsample(midlevel_features)
        return output

    def train(model, trainloader, valloader, num_epoch=10):
        print("Start training...")
        trn_loss_hist = []
        trn_acc_hist = []
        val_acc_hist = []

        model.train()  # Set the model to training mode

        for i in range(num_epoch):
            running_loss = []
            print('-----------------Epoch = %d-----------------' % (i + 1))
            for (gray_in, color_ab_in) in tqdm(trainloader):
                gray_in_Var = Variable(gray_in).cuda() if use_gpu else Variable(input_gray)
                color_ab_in_Var = Variable(color_ab_in).cuda() if use_gpu else Variable(input_ab)

                optimizer.zero_grad()
                pred = model(gray_in_Var)
                # Calculate the loss
                loss = criterion(pred, color_ab_in_Var)
                # Record loss and accuracy
                running_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            print("\n Epoch {} loss:{}".format(i + 1, np.mean(running_loss)))

            # Keep track of training loss, accuracy, and validation loss
            trn_loss_hist.append(np.mean(running_loss))
            trn_acc_hist.append(evaluate(model, trainloader))
            print("\n Evaluate on validation set...")
            val_acc_hist.append(evaluate(model, valloader))
        print("Done!")
        return trn_loss_hist, trn_acc_hist, val_acc_hist

    def evaluate(model, loader):
        # Set the model to evaluation mode
        model.eval()
        correct = 0
        with torch.no_grad():
            for (gray_in, color_ab_in) in tqdm(loader):
                gray_in_Var = Variable(gray_in).cuda() if use_gpu else Variable(input_gray)
                color_ab_in_Var = Variable(color_ab_in).cuda() if use_gpu else Variable(input_ab)
                pred = model(gray_in_Var)
                correct += (torch.argmax(pred, dim=1) == color_ab_in_Var).sum().item()
            acc = correct / len(loader.dataset)
            print("\n Evaluation accuracy: {}".format(acc))
            return acc
