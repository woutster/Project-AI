from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

import torch.nn as nn
import torch
import pandas as pd
import numpy as np


class LSTM(nn.Module):

    def __init__(self, batch_size, input_size,  output_size,
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

    X, Y_pos, Y_neg = get_data(f'Data/{args.bus}/{args.bus}_merged_data.csv', args.batch_size)

    input_size = X.shape[2]

    # Initialise model
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
    pos_criterion = torch.nn.MSELoss().to(device)
    pos_optimizer = torch.optim.RMSprop(pos_model.parameters(), lr=args.learning_rate)

    neg_criterion = torch.nn.MSELoss().to(device)
    neg_optimizer = torch.optim.RMSprop(pos_model.parameters(), lr=args.learning_rate)

    # Iterate over data
    for step, batch_inputs in enumerate(X):


        # TODO: reshape data?
        x = batch_inputs.view(1, args.batch_size, input_size)

        pos_optimizer.zero_grad()
        neg_optimizer.zero_grad()

        p_out, (ph,pc) = pos_model(x)
        n_out, (nh, nc) = neg_model(x)
        # p_acc = accuracy(p_out, Y_pos[step], args.batch_size)
        # n_acc = accuracy(n_out, Y_neg[step], args.batch_size)

        p_loss = pos_criterion(p_out.transpose(0,1), Y_pos[step].view(args.batch_size,1))

        p_loss.backward()
        pos_optimizer.step()

        n_loss = neg_criterion(n_out.transpose(0,1), Y_neg[step].view(args.batch_size, 1))

        n_loss.backward()
        neg_optimizer.step()

        if step % args.eval_every == 0:
            # import pdb; pdb.set_trace()
            print("Training step: ", step)
            print("Pos loss: ", p_loss.item())
            print("Neg loss: ", n_loss.item())


def accuracy(predictions, target, batch_size):
    prediction = predictions.argmax(dim=2)
    target = targets

    accuracy = (target == prediction).float().sum() / batch_size

    return accuracy.item()


def process_data(df):
    df.dropna(inplace=True)
    df = df.drop(['gps_point_entry', 'gps_point_exit'], axis=1).astype(np.float)
    return df


def get_data(filename, batch_size):
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
    return make_batches(df, batch_size)


def make_batches(df, batch_size):
    x, ypos, yneg = [], [], []
    x_batch, ypos_batch, yneg_batch = [], [], []

    cmf = pd.DataFrame()
    cmf['pos'] = df['cmf_pos']
    cmf['neg'] = df['cmf_neg']
    del df['cmf_pos']
    del df['cmf_neg']

    iter = 0
    lenin = len(df)
    for i in range(lenin):
        if i + batch_size >= lenin:
            break
        for j in range(batch_size):
            x_batch.append(df.iloc[i + j].values)
            ypos_batch.append(cmf['pos'].iloc[i + j])
            yneg_batch.append(cmf['neg'].iloc[i + j])
        x.append(x_batch)
        ypos.append(ypos_batch)
        yneg.append(yneg_batch)
        x_batch, ypos_batch, yneg_batch = [], [], []

    return torch.tensor(x).type(torch.FloatTensor), torch.tensor(ypos).type(torch.FloatTensor), torch.tensor(yneg).type(torch.FloatTensor)


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    # parser.add_argument('--output_seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(1e6), help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')
    parser.add_argument('--eval_every', type=float, default=100, help='--')

    # Bus specific
    parser.add_argument('--bus', type=str, default='proov_001', help='Bus to train data on.')

    args = parser.parse_args()

    # Train the model
    train(args)
