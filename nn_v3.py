import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Subset, Dataset, DataLoader
from datetime import datetime
import time
import pandas as pd
import numpy as np


class TraficDataset(Dataset):
    """Trafic dataset
        We return as input data the three hours after the given index (so 180 * 4 floats)
        We return as target the average values for the time t+30, t+90, t+150
    ."""

    def __init__(self, csv_file, timeshift=150, transform=None, shuffle=False):
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
        if shuffle:
            self.trafic = self.trafic.sample(frac=1)

    def __len__(self):
        # Since we need to predict for the next 3 hours, we do not include the last point in the training data
        # Also, for training, the idx is the index of the last time point, we check the 3 hours before (180 points)
        return len(self.trafic)-500
        # return len(self.trafic) // 10  # For testing only

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = self.trafic.loc[idx, :].values.flatten().tolist()
        outputs = inputs[-6:]
        inputs = inputs[:-6]

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
        if batch % 25 == 24:
            print(f'Train avg_train_loss: {sum(losses) / (batch + 1)}')

    return losses


def test(test_dataloader, model: torch.nn.Module, criterion: torch.nn.MSELoss):
    model.eval()
    losses = []

    with torch.no_grad():
        for batch, data in enumerate(test_dataloader):
            pred = model(data["inputs"]).float()
            loss = criterion(pred, data["outputs"])
            loss = loss.float()

            losses.append(loss.detach().numpy())
            # if batch % 500 == 499:
            #    print(f'Test avg_test_loss: {sum(losses) / (batch + 1)}')
    print(f'Full test avg_test_loss: {sum(losses) / (batch + 1)}')
    return losses


def train_nn(csv_file):
    dataset = TraficDataset(csv_file)

    # Define the size of the training set (e.g., 80%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # Create indices for the training and testing subsets
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, len(dataset)))

    # Create Subset datasets for train and test
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False, num_workers=8)

    sample = train_dataset[0]
    input_dim = len(sample["inputs"])
    output_dim = len(sample["outputs"])
    model = FNN(input_dim=input_dim, output_dim=output_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)  # 0.0001
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.1,
        patience=5,
    )

    train_losses, test_losses = [], []
    start_time = time.time()
    for epoch in range(20):
        start_epoch = time.time()
        print('epoch:', epoch)
        train_loss = train(train_loader, model, criterion, optimizer)
        train_losses.append(train_loss)

        test_loss = test(test_loader, model, criterion)
        test_losses.append(test_loss)

        print(f"Epoch {epoch+1} training time: {time.time()-start_epoch:.2f}")
        scheduler.step(np.mean(train_losses))
        torch.save(model.state_dict(), f'/home/yohan/Documents/dataset/trained_nn/nn_v3_epoch_{epoch}.pth')

    # test_loss = test(test_loader, model, criterion)
    # print(f'avg_test_loss: {sum(test_loss) / test_size}')
    print(f"Full training time: {time.time() - start_time:.2f}")


def check_data_point(dataset=None):
    if dataset is None:
        dataset = TraficDataset(csv_file="/home/yohan/Documents/dataset/HistoricalData/CH:0056.05_nn_ready.csv")
    sample = dataset[5000]
    # print(sample)
    print(len(sample["inputs"]))
    print(len(sample["outputs"]))


if __name__ == '__main__':
    # check_data_point()
    train_nn(csv_file="/home/yohan/Documents/dataset/HistoricalData/CH:0056.05_nn_ready.csv")

