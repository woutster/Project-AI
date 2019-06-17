from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

import torch.nn as nn
import torch
import pandas as pd
import numpy as np


class LSTM(nn.Module):

    def __init__(self, batch_size, input_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda'):

        super(LSTM, self).__init__()

        # Initialization of LSTM
        self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=lstm_num_hidden,
                    num_layers=lstm_num_layers,
                    bidirectional=False,
                    batch_first=True
        )

        self.linear = nn.Linear(lstm_num_hidden, input_size, bias=True)

    def forward(self, x, hc = None):
        """Forward pass of LSTM"""
        out, (h, c) = self.lstm(x, hc)

        out = self.linear(out)

        return out, (h,c)


def train(args):

    # Initialize the device which to run the model on
    if args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device(args.device)
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    # TODO: data preproccessing

    X, Y = get_data(f'Data/{args.bus}/{args.bus}_merged_data.csv')
    # del data['in_time']
    # del data['out_time']
    # del data['time']
    # del data['date']
    print(X.columns)

    print(Y.columns)
    input_size = len(data.columns)

    # Initialise model
    model = LSTM(batch_size=args.batch_size,
                input_size=input_size - 2,
                lstm_num_hidden=args.lstm_num_hidden,
                lstm_num_layers=args.lstm_num_layers,
                device=device
    )
    model.to(device)

    # Set up the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)

    # Iterate over data
    for step, datapoint in X.iterrows():
        print(datapoint.values.dtype)
        # TODO: reshape data?
        x = list(datapoint.values)
        print(x)
        x = torch.tensor(x[2:]).view(1,1,-1)
        optimizer.zero_grad()
        out, (h,c) = model(x)

        # TODO: not reshape data?
        loss = criterion(out.transpose(2,1), y)
        loss.backward()
        optimizer.step()


def process_data(df):
    df.dropna(inplace=True)
    df = df.drop(['gps_point_entry', 'gps_point_exit'], axis=1).astype(np.float)
    return df


def get_data(filename):
    df = pd.read_csv(filename, sep=';')

    # print(df.columns)
    df['timestamp_in'] = df['Unnamed: 0']
    del df['Unnamed: 0']

    df['timestamp_in'] = pd.to_datetime(df['timestamp_in'], format='%Y-%m-%d %H:%M:%S')
    df['timestamp_in'] = df.timestamp_in.values.astype(np.float64) // 10 ** 9

    df['timestamp_exit'] = pd.to_datetime(df['timestamp_exit'], format='%Y-%m-%d %H:%M:%S')
    df['timestamp_exit'] = df.timestamp_exit.values.astype(np.float64) // 10 ** 9
    df = process_data(df)
    # return df
    return make_targets(df)


def make_targets(df):
    lenin = df.shape[0]

    x, y = [], []
    # print(df['cmf_pos'])

    cmf = pd.DataFrame()
    cmf['pos'] = df['cmf_pos']
    cmf['neg'] = df['cmf_neg']
    # del df['cmf_pos']
    # del df['cmf_neg']
    import math
    for i, row in df.iterrows():
        if i == lenin:
            break
        if math.isnan(row['cmf_neg']):
            continue

        x.append(row.values[2:])
        y.append((cmf['pos'].iloc[i], cmf['neg'].iloc[i]))
    x = np.array(x[:-100])
    y = np.array(y[:-100])
    # print(y)
    # print(x)

    return torch.from_numpy(x), torch.from_numpy(y)


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    # parser.add_argument('--output_seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(1e6), help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Bus specific
    parser.add_argument('--bus', type=str, default='proov_001', help='Bus to train data on.')

    args = parser.parse_args()

    # Train the model
    train(args)
