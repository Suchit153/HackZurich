import os

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import torch.nn as nn
import tqdm


class TraficDataset(Dataset):
    """Trafic dataset
        We return as input data the three hours after the given index (so 180 * 4 floats)
        We return as target the next 5,10,15,...,175,180 minutes after the last input point
    ."""

    def __init__(self, csv_file, timeshift=150, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            timeshift: The number of minute after the end of the input sample for the target
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.trafic = pd.read_csv(csv_file)
        self.transform = transform
        self.timeshift = timeshift
        self.date_format = '%Y-%m-%dT%H:%M:%S.000000Z'

    def __len__(self):
        # Since we need to predict for the next 3 hours, we do not include the last point in the training data
        # Also, for training, the idx is the index of the last time point, we check the 3 hours before (180 points)
        # return len(self.trafic)-500
        return len(self.trafic)//10  # For testing onl

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = []
        outputs = []
        for i in range(180):
            # The timestamp is an issue, the number is way to big now
            date = datetime.strptime(self.trafic.loc[i+idx, "TimeStamp"], self.date_format)
            inputs.append(float(date.weekday())/6.0)
            inputs.append(float(date.hour)/23.0)
            inputs.append(float(date.minute)/60.0)
            inputs.append(float(self.trafic.loc[i+idx, "CarFlow"])/1500.0)
            inputs.append(float(self.trafic.loc[i+idx, "CarSpeed"])/130.0)

        outputs.append(float(self.trafic.loc[idx+180+self.timeshift, "CarFlow"])/1000.0)
        outputs.append(float(self.trafic.loc[idx+180+self.timeshift, "CarSpeed"])/130.0)
        inputs = torch.FloatTensor(inputs)
        outputs = torch.FloatTensor(outputs)
        sample = {'inputs': inputs, 'outputs': outputs}

        return sample


class FNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FNN, self).__init__()

        self.linear1 = nn.Linear(input_dim, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 256)
        self.linear4 = nn.Linear(256, 64)
        self.linear5 = nn.Linear(64, output_dim)

        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.ReLU(self.linear1(x))
        x = self.ReLU(self.linear2(x))
        x = self.ReLU(self.linear3(x))
        x = self.ReLU(self.linear4(x))
        x = self.linear5(x)
        return x


def check_data_point(dataset=None):
    if dataset is None:
        dataset = TraficDataset(csv_file="/home/yohan/Documents/dataset/HistoricalData/CH:0056.05_interpolated_v2.csv")
    sample = dataset[5000]
    print(sample)
    print(len(sample["inputs"]))
    pass


def train(train_dataloader, model: torch.nn.Module, criterion: torch.nn.MSELoss, optimizer: torch.optim.Adam):
    model.train()
    losses = []

    for batch, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        pred = model(data["inputs"]).float()

        loss = criterion(pred, data["outputs"])

        loss.backward()

        optimizer.step()

        losses.append(loss.detach())

        print(f'avg_loss: {sum(losses)/(batch+1)}')

    return losses


def test(test_dataloader, model: torch.nn.Module, criterion: torch.nn.MSELoss):
    model.eval()
    losses = []

    with torch.no_grad():
        for batch, data in enumerate(test_dataloader):
            x, y = data

            x = x.float()
            y = y.float()

            pred = model(x).float()

            loss = criterion(pred, y)

            losses.append(loss.detach().numpy())
            if batch % 100 == 99:
                print(f'avg_loss: {sum(losses) / (batch + 1)}')

    return losses


def train_nn():
    dataset = TraficDataset(csv_file="/home/yohan/Documents/dataset/HistoricalData/CH:0056.05_interpolated_v2.csv")
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True, num_workers=4)

    model = FNN(input_dim=900, output_dim=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)  # 0.0001
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.1,
        patience=5,
    )

    train_losses = []

    for epoch in range(5):
        train_loss = train(dataloader, model, criterion, optimizer)
        train_losses.append(train_loss)
        print(f"\nTrain losses\n{train_loss}")
        print(f"\nTrain losses\n{len(train_loss)}")
        scheduler.step(np.mean(train_losses))
        torch.save(model.state_dict(), f'/home/yohan/Documents/dataset/epoch_{epoch}.pth')


if __name__ == '__main__':
    train_nn()
    # check_data_point()

    pass