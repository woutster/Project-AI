from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

class BusDataset(data.Dataset):

    def __init__(self, bus, train):
        if train:
            self.df = pd.read_csv(f'Data/proov_001/proov_001_merged_data.csv', sep=';')
            self.df2 = pd.read_csv(f'Data/proov_002/proov_002_merged_data.csv', sep=';')
            self.df = self.df.append(self.df2, ignore_index=True)
        else:
            self.df = pd.read_csv(f'Data/proov_003/proov_003_merged_data.csv', sep=';')


        # Maybe remove columns
        self.df['timestamp_in'] = self.df['Unnamed: 0']
        del self.df['Unnamed: 0']

        self.df['timestamp_in'] = pd.to_datetime(self.df['timestamp_in'], format='%Y-%m-%d %H:%M:%S')
        self.df['timestamp_in'] = self.df.timestamp_in.values.astype(np.float64) // 10 ** 9

        self.df['timestamp_exit'] = pd.to_datetime(self.df['timestamp_exit'], format='%Y-%m-%d %H:%M:%S')
        self.df['timestamp_exit'] = self.df.timestamp_exit.values.astype(np.float64) // 10 ** 9

        self.df = self.process_data(self.df)

        self.cmf = pd.DataFrame()
        self.cmf['pos'] = self.df['cmf_pos']
        self.cmf['neg'] = self.df['cmf_neg']
        del self.df['cmf_pos']
        del self.df['cmf_neg']

        self.data_size = self.df.shape[0]

    def process_data(self, df):
        df.dropna(inplace=True)
        df = df.drop(['gps_point_entry', 'gps_point_exit'], axis=1).astype(np.float)
        return df

    def __getitem__(self, item):
        inputs = torch.tensor(self.df.iloc[item].values).type(torch.FloatTensor)
        pos_targets = torch.tensor(self.cmf['pos'].iloc[item]).type(torch.FloatTensor)
        neg_targets = torch.tensor(self.cmf['neg'].iloc[item]).type(torch.FloatTensor)
        return inputs, pos_targets, neg_targets

    def __len__(self):
        return self.data_size

    @property
    def input_size(self):
        return self.df.shape[1]


class LSTM(nn.Module):

    def __init__(self, batch_size, input_size, output_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda'):
        super(LSTM, self).__init__()

        # Initialization of LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_num_hidden,
            num_layers=lstm_num_layers,
            bidirectional=False,
            batch_first=False
        )

        self.linear = nn.Linear(lstm_num_hidden, output_size, bias=True)

    def forward(self, x, hc=None):
        """Forward pass of LSTM"""
        out, (h, c) = self.lstm(x, hc)

        out = self.linear(out)

        return out, (h, c)


def make_plots(pos_pred, pos_train, neg_pred, neg_train):
    pos_targets, pos_out = pos_pred
    plt.subplot(2, 2, 1)
    plt.plot(pos_targets, color='r', label='Targets')
    plt.plot(pos_out, color='b', label='Predictions')
    plt.title("Positive cmf targets vs the predicted positive cmf")
    plt.legend()

    pos_loss, pos_acc = pos_train
    plt.subplot(2, 2, 2)
    plt.plot(pos_loss, color='r', label='Loss')
    plt.plot(pos_acc, color='b', label='Accuracy')
    plt.title("Loss and accuracy of the trained positive model")
    plt.legend()

    neg_targets, neg_out = neg_pred
    plt.subplot(2, 2, 3)
    plt.plot(neg_targets, color='r', label='Targets')
    plt.plot(neg_out, color='b', label='Predictions')
    plt.title("Negative cmf targets vs the predicted negative cmf")
    plt.legend()

    neg_loss, neg_acc = neg_train
    plt.subplot(2, 2, 4)
    plt.plot(neg_loss, color='r', label='Loss')
    plt.plot(neg_acc, color='b', label='Accuracy')
    plt.title("Loss and accuracy of the trained negative model")
    plt.legend()

    plt.show()


def accuracy(predictions, target, batch_size, tolerance):
    prediction = predictions.detach().numpy()[0].T
    target = target.numpy()

    diff_array = np.abs(prediction - target)
    return ((diff_array <= tolerance).sum()/batch_size) * 100

def train(args):
    # Initialize the device which to run the model on
    if args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device(args.device)
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    # Load data
    train_bus_data = BusDataset('proov_00*', train=True)
    train_data_loader = DataLoader(train_bus_data, batch_size=args.batch_size)
    test_bus_data = BusDataset('proov_00*', train=False)
    test_data_loader = DataLoader(test_bus_data, batch_size=len(test_bus_data))

    input_size = train_bus_data.input_size

    # Initialise models
    pos_model = LSTM(batch_size=args.batch_size,
                     input_size=input_size,
                     lstm_num_hidden=args.lstm_num_hidden,
                     lstm_num_layers=args.lstm_num_layers,
                     device=device,
                     output_size=1
                     )
    pos_model.to(device)

    neg_model = LSTM(batch_size=args.batch_size,
                     input_size=input_size,
                     lstm_num_hidden=args.lstm_num_hidden,
                     lstm_num_layers=args.lstm_num_layers,
                     device=device,
                     output_size=1
                     )
    neg_model.to(device)

    # Set up the loss and optimizer
    pos_criterion = torch.nn.L1Loss().to(device)
    pos_optimizer = torch.optim.Adam(pos_model.parameters(), lr=args.learning_rate)

    neg_criterion = torch.nn.L1Loss().to(device)
    neg_optimizer = torch.optim.Adam(neg_model.parameters(), lr=args.learning_rate)

    # Plotting prep
    all_pos_targets = []
    all_pos_outs = []
    all_pos_losses = []
    all_pos_accuracies = []
    all_neg_targets = []
    all_neg_outs = []
    all_neg_losses = []
    all_neg_accuracies = []

    # Iterate over data
    for step, (batch_inputs, pos_targets, neg_targets) in enumerate(train_data_loader):

        # print(batch_inputs.shape)
        if batch_inputs.shape[0] != args.batch_size:
            continue

        x = batch_inputs.view(1, args.batch_size, input_size)

        pos_optimizer.zero_grad()
        neg_optimizer.zero_grad()

        p_out, _ = pos_model(x)
        n_out, _ = neg_model(x)

        p_loss = pos_criterion(p_out.transpose(0, 1), pos_targets.view(args.batch_size, 1))
        p_loss.backward()
        pos_optimizer.step()

        # import pdb; pdb.set_trace()
        n_loss = neg_criterion(n_out.transpose(0, 1), neg_targets.view(args.batch_size, 1))
        n_loss.backward()
        neg_optimizer.step()

        p_acc = accuracy(p_out, pos_targets, args.batch_size, args.acc_bound)
        n_acc = accuracy(n_out, neg_targets, args.batch_size, args.acc_bound)

        # Plotting statistics
        all_pos_targets.extend(pos_targets.tolist())
        all_pos_outs.extend(p_out.view(args.batch_size).tolist())
        all_pos_losses.append(p_loss.item())
        all_pos_accuracies.append(p_acc)
        all_neg_targets.extend(neg_targets.tolist())
        all_neg_outs.extend(n_out.view(args.batch_size).tolist())
        all_neg_losses.append(n_loss.item())
        all_neg_accuracies.append(n_acc)

        if step % args.eval_every == 0:
            p_acc = accuracy(p_out, pos_targets, args.batch_size, args.acc_bound)
            n_acc = accuracy(n_out, neg_targets, args.batch_size, args.acc_bound)

            print("Training step: ", step)
            print("Pos loss: ", p_loss.item())
            print("Neg loss: ", n_loss.item())
            print("Pos acc: ", p_acc, "%")
            print("Neg acc:", n_acc, "%")
            print('')

        if step == 2500:
            break

    print("Done training...\n")
    print("Testing...")

    for step, (test_inputs, pos_targets, neg_targets) in enumerate(test_data_loader):

        x = test_inputs.view(1, test_inputs.shape[0], input_size)

        import pdb; pdb.set_trace()

        pos_optimizer.zero_grad()
        neg_optimizer.zero_grad()

        p_test_out, _ = pos_model(x)
        n_test_out, _ = neg_model(x)

        p_test_loss = pos_criterion(p_test_out.transpose(0, 1), pos_targets.view(pos_targets.shape[0], 1))
        p_test_acc = accuracy(p_test_out, pos_targets, len(p_test_out[0]), args.acc_bound)
        n_test_loss = pos_criterion(n_test_out.transpose(0, 1), neg_targets.view(neg_targets.shape[0], 1))
        n_test_acc = accuracy(n_test_out, neg_targets, len(n_test_out[0]), args.acc_bound)

    print("Pos loss:", p_test_loss.item())
    print("Neg loss: ", n_test_loss.item())
    print("Pos acc:", p_test_acc, "%")
    print("Neg acc:", n_test_acc, "%")
    print('')
    make_plots((all_pos_targets, all_pos_outs), (all_pos_losses, all_pos_accuracies), (all_neg_targets, all_neg_outs), (all_neg_losses, all_neg_accuracies))



if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    # parser.add_argument('--output_seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(1e6), help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')
    parser.add_argument('--eval_every', type=float, default=500, help='--')
    parser.add_argument('--acc_bound', type=float, default=1, help='--')

    # Bus specific
    parser.add_argument('--bus', type=str, default='proov_001', help='Bus to train data on.')

    args = parser.parse_args()

    # Train the model
    train(args)
