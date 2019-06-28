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


class BusDataset(data.Dataset):

    def __init__(self, bus, train):
        if train:
            self.df = pd.read_csv(f'Data/proov_001/proov_001_merged_data.csv', sep=';')
            self.df2 = pd.read_csv(f'Data/proov_003/proov_003_merged_data.csv', sep=';')
            self.df = self.df.append(self.df2, ignore_index=True)
        else:
            self.df = pd.read_csv(f'Data/proov_002/proov_002_merged_data.csv', sep=';')

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
        df = df.drop(['gps_point_entry', 'gps_point_exit', 'timestamp_in', 'timestamp_exit'], axis=1).astype(np.float)
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

    def __init__(self, batch_size, input_size, output_size, dropout,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda'):
        super(LSTM, self).__init__()

        # Initialization of LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_num_hidden,
            num_layers=lstm_num_layers,
            bidirectional=False,
            batch_first=False,
            dropout=dropout
        )

        self.linear = nn.Linear(lstm_num_hidden, output_size, bias=True)

    def forward(self, x, hc=None):
        """Forward pass of LSTM"""
        out, (h, c) = self.lstm(x, hc)

        out = self.linear(out)

        return out, (h, c)


def make_plots(pos_pred, pos_train, neg_pred, neg_train, pos_test, neg_test):
    pos_targets, pos_out = pos_pred
    plt.subplot(2, 2, 1)
    plt.plot(pos_targets, color='r', label='Targets')
    plt.plot(pos_out, color='b', label='Predictions')
    plt.xlabel('Datapoint from 2017')
    plt.ylabel('CMF score')
    plt.title("Positive cmf targets vs the predicted positive cmf")
    plt.legend()

    pos_loss, pos_acc = pos_train
    plt.subplot(2, 2, 2)
    plt.plot(pos_loss, color='r', label='Loss')
    plt.plot(pos_acc, color='b', label='Accuracy')
    plt.xlabel('Iteration number')
    plt.ylabel('Loss and accuracy (in percentages) score')
    plt.title("Loss and accuracy of the trained positive model")
    plt.legend()

    neg_targets, neg_out = neg_pred
    plt.subplot(2, 2, 3)
    plt.plot(neg_targets, color='r', label='Targets')
    plt.plot(neg_out, color='b', label='Predictions')
    plt.xlabel('Datapoint from 2017')
    plt.ylabel('CMF score')
    plt.title("Negative cmf targets vs the predicted negative cmf")
    plt.legend()

    neg_loss, neg_acc = neg_train
    plt.subplot(2, 2, 4)
    plt.plot(neg_loss, color='r', label='Loss')
    plt.plot(neg_acc, color='b', label='Accuracy')
    plt.xlabel('Iteration number')
    plt.ylabel('Loss and accuracy (in percentages) score')
    plt.title("Loss and accuracy of the trained negative model")
    plt.legend()

    plt.show()

    plt.subplot(2,1,1)
    pred_pos, target_pos = pos_test
    pred_pos = np.array(pred_pos).ravel()
    target_pos = target_pos.numpy()
    plt.plot(target_pos, color='b', label='target')
    plt.plot(pred_pos, color='r', label='prediction')
    plt.xlabel('Datapoint from 2017')
    plt.ylabel('CMF score')
    plt.title("Prediction vs target test set of positive cmf")
    plt.legend()

    plt.subplot(2, 1, 2)
    pred_neg, target_neg = neg_test
    pred_neg = np.array(pred_neg).ravel()
    target_neg = target_neg.numpy()
    plt.plot(target_neg, color='b', label='target')
    plt.plot(pred_neg, color='r', label='prediction')
    plt.xlabel('Datapoint from 2017')
    plt.ylabel('CMF score')
    plt.title("Prediction vs target test set of negative cmf")
    plt.legend()

    plt.show()


def accuracy(predictions, target, batch_size, tolerance):
    prediction = predictions.detach().numpy()[0].T
    target = target.numpy()

    diff_array = np.abs(prediction - target)
    return ((diff_array <= tolerance).sum()/batch_size) * 100


def get_baseline(pos_pred, neg_pred):
    df = pd.read_csv(f'Data/proov_002/proov_002_merged_data.csv', sep=';')

    df['timestamp_in'] = df['Unnamed: 0']
    del df['Unnamed: 0']

    df['timestamp_in'] = pd.to_datetime(df['timestamp_in'], format='%Y-%m-%d %H:%M:%S')
    df['timestamp_exit'] = pd.to_datetime(df['timestamp_exit'], format='%Y-%m-%d %H:%M:%S')

    df.dropna(inplace=True)
    cmf = pd.DataFrame()
    cmf['pos'] = df['cmf_pos']
    cmf['neg'] = df['cmf_neg']
    cmf['timestamp'] = df['timestamp_in']
    cmf = cmf.set_index('timestamp')
    davg = cmf.resample('D').mean().fillna(method='ffill')
    davg['pos'] = davg['pos'].shift(1)
    davg['neg'] = davg['neg'].shift(1)
    davg = davg.fillna(method='bfill')

    cmf.index = cmf.index.normalize()

    total = []

    for index, row in davg.iterrows():
        if index in cmf.index:
            daily_values_pos = cmf.get_value(index, 'pos')
            daily_values_neg = cmf.get_value(index, 'neg')
            if not hasattr(daily_values_neg, "__iter__"):
                daily_values_neg = [daily_values_neg]
                daily_values_pos = [daily_values_pos]
            for i, _ in enumerate(daily_values_pos):
                total.append([daily_values_pos[i], daily_values_neg[i], row['pos'], row['neg']])
        else:
            continue

    array = np.array(total)
    diffs_pos = np.abs(array[:, 0] - array[:, 2])
    diffs_neg = np.abs(array[:, 1] - array[:, 3])
    len_pos = len(np.where(diffs_pos < 0.5)[0])
    len_neg = len(np.where(diffs_neg < 0.5)[0])

    return array


def train(args, flag):
    # Initialize the device which to run the model on
    if flag == 'terminal':
        if args.device == 'cuda':
            if torch.cuda.is_available():
                device = torch.device(args.device)
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(args.device)

        batch_size = args.batch_size
        lstm_num_hidden = args.lstm_num_hidden
        lstm_num_layers = args.lstm_num_layers
        dropout_keep_prob = args.dropout_keep_prob
        learning_rate = args.learning_rate
        train_steps = args.train_steps
        acc_bound = args.acc_bound
        eval_every = args.eval_every

    elif flag == 'tuning':
        device = args[0]
        batch_size = args[1]
        lstm_num_hidden = args[2]
        lstm_num_layers = args[3]
        dropout_keep_prob = args[4]
        learning_rate = args[5]
        train_steps = args[6]
        acc_bound = args[7]
        eval_every = args[8]

    # Load data
    train_bus_data = BusDataset('proov_00*', train=True)
    train_data_loader = DataLoader(train_bus_data, batch_size=batch_size)
    test_bus_data = BusDataset('proov_00*', train=False)
    test_data_loader = DataLoader(test_bus_data, batch_size=len(test_bus_data))

    input_size = train_bus_data.input_size

    # Initialise models
    pos_model = LSTM(batch_size=batch_size,
                     input_size=input_size,
                     lstm_num_hidden=lstm_num_hidden,
                     lstm_num_layers=lstm_num_layers,
                     device=device,
                     output_size=1,
                     dropout=dropout_keep_prob
                     )
    pos_model.to(device)

    neg_model = LSTM(batch_size=batch_size,
                     input_size=input_size,
                     lstm_num_hidden=lstm_num_hidden,
                     lstm_num_layers=lstm_num_layers,
                     device=device,
                     output_size=1,
                     dropout=dropout_keep_prob
                     )
    neg_model.to(device)

    # Set up the loss and optimizer
    pos_criterion = torch.nn.MSELoss().to(device)
    pos_optimizer = torch.optim.RMSprop(pos_model.parameters(), lr=learning_rate)

    neg_criterion = torch.nn.MSELoss().to(device)
    neg_optimizer = torch.optim.RMSprop(neg_model.parameters(), lr=learning_rate)

    # Iterate over data
    for epoch in range(0, train_steps):
        # Plotting prep
        all_pos_targets = []
        all_pos_outs = []
        all_pos_losses = []
        all_pos_accuracies = []
        all_neg_targets = []
        all_neg_outs = []
        all_neg_losses = []
        all_neg_accuracies = []
        print("Training epoch", epoch)
        for step, (batch_inputs, pos_targets, neg_targets) in enumerate(train_data_loader):

            # print(batch_inputs.shape)
            if batch_inputs.shape[0] != batch_size:
                continue

            x = batch_inputs.view(1, batch_size, input_size)

            pos_optimizer.zero_grad()
            neg_optimizer.zero_grad()

            p_out, _ = pos_model(x)
            n_out, _ = neg_model(x)

            p_loss = pos_criterion(p_out.transpose(0, 1), pos_targets.view(batch_size, 1))
            p_loss.backward()
            pos_optimizer.step()

            # import pdb; pdb.set_trace()
            n_loss = neg_criterion(n_out.transpose(0, 1), neg_targets.view(batch_size, 1))
            n_loss.backward()
            neg_optimizer.step()

            p_acc = accuracy(p_out, pos_targets, batch_size, acc_bound)
            n_acc = accuracy(n_out, neg_targets, batch_size, acc_bound)

            # Plotting statistics
            all_pos_targets.extend(pos_targets.tolist())
            all_pos_outs.extend(p_out.view(batch_size).tolist())
            all_pos_losses.append(p_loss.item())
            all_pos_accuracies.append(p_acc)
            all_neg_targets.extend(neg_targets.tolist())
            all_neg_outs.extend(n_out.view(batch_size).tolist())
            all_neg_losses.append(n_loss.item())
            all_neg_accuracies.append(n_acc)

            if step % eval_every == 0:
                p_acc = accuracy(p_out, pos_targets, batch_size, acc_bound)
                n_acc = accuracy(n_out, neg_targets, batch_size, acc_bound)

                print("Training step: ", step)
                print("Pos loss: ", p_loss.item())
                print("Neg loss: ", n_loss.item())
                print("Pos acc: ", p_acc, "%")
                print("Neg acc:", n_acc, "%")
        print('----')

    print("Done training...\n")
    print("Testing...")

    pos_pred_test = []
    neg_pred_test = []

    for epoch, (test_inputs, pos_targets, neg_targets) in enumerate(test_data_loader):

        for _, x in enumerate(test_inputs):

            pos_optimizer.zero_grad()
            neg_optimizer.zero_grad()

            x = x.view(1, 1, input_size)
            p_test_out, _ = pos_model(x)
            n_test_out, _ = neg_model(x)

            pos_pred_test.append(p_test_out.detach().numpy()[0])
            neg_pred_test.append(n_test_out.detach().numpy()[0])

    pos_pred_test_torch = torch.from_numpy(np.array(pos_pred_test))
    neg_pred_test_torch = torch.from_numpy(np.array(neg_pred_test))

    p_test_loss = pos_criterion(pos_pred_test_torch.transpose(0, 1), pos_targets.view(pos_targets.shape[0], 1))
    p_test_acc = accuracy(pos_pred_test_torch.transpose(0, 1), pos_targets, len(pos_pred_test), acc_bound)
    n_test_loss = pos_criterion(neg_pred_test_torch.transpose(0, 1), neg_targets.view(neg_targets.shape[0], 1))
    n_test_acc = accuracy(neg_pred_test_torch.transpose(0, 1), neg_targets, len(neg_pred_test), acc_bound)

    print("Pos loss:", p_test_loss.item())
    print("Neg loss: ", n_test_loss.item())
    print("Pos acc:", p_test_acc, "%")
    print("Neg acc:", n_test_acc, "%")
    print('')

    get_baseline(np.array(pos_pred_test).ravel(), np.array(neg_pred_test).ravel())
    if flag == 'terminal':
        make_plots((all_pos_targets, all_pos_outs), (all_pos_losses, all_pos_accuracies), (all_neg_targets, all_neg_outs), (all_neg_losses, all_neg_accuracies), (pos_pred_test, pos_targets), (neg_pred_test, neg_targets))
    elif flag == 'tuning':
        return p_test_acc, n_test_acc


def tune_hyperparameters(flag):
    batch_size_loop = [16, 32, 64]
    lstm_num_hidden_loop = [1, 16, 64, 128]
    lstm_num_layers_loop = [1, 2, 4, 8, 16]
    # dropout_keep_prob_loop = [0, 0.25, 0.5]
    learning_rate_loop = [1e-1, 1e-2, 1e-3, 1e-4]
    device = 'cpu'
    train_steps = int(1)
    eval_every = 200
    acc_bound = 0.5
    dropout_keep_prob = 0
    acc = []
    acc_dict = {}
    steps = len(batch_size_loop) * len(lstm_num_hidden_loop) * len(lstm_num_layers_loop) * len(learning_rate_loop)
    counter = 0
    for batch_size in batch_size_loop:
        for lstm_num_hidden in lstm_num_hidden_loop:
            for lstm_num_layers in lstm_num_layers_loop:
                # for dropout_keep_prob in dropout_keep_prob_loop:
                for learning_rate in learning_rate_loop:
                    acc_dict[counter] = [batch_size, lstm_num_hidden, lstm_num_layers, learning_rate]
                    counter += 1
                    print('Training model', counter, '/', steps, 'with:')
                    print('- batch size: ', batch_size)
                    print('- lstm_num_hidden: ', lstm_num_hidden)
                    print('- lstm_num_layers: ', lstm_num_layers)
                    # print('- dropout_keep_prob: ', dropout_keep_prob)
                    print('- learning rate: ', learning_rate)
                    print('-------------')
                    print('')
                    p_acc, n_acc = train(
                        [device, batch_size, lstm_num_hidden, lstm_num_layers, dropout_keep_prob, learning_rate,
                         train_steps, acc_bound, eval_every], flag)
                    acc.append([p_acc, n_acc, p_acc + n_acc])
    import pdb;
    pdb.set_trace()


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_type', type=str, default='terminal', help='Tune the parameters or set your own')

    # Model params
    parser.add_argument('--lstm_num_hidden', type=int, default=64, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=8, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=32, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on')

    # It is not necessary to implement the following three params, but it may help training.
    # parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    # parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(1), help='Number of training steps')
    parser.add_argument('--eval_every', type=float, default=100, help='--')
    parser.add_argument('--acc_bound', type=float, default=0.5, help='--')

    # Bus specific
    # parser.add_argument('--bus', type=str, default='proov_001', help='Bus to train data on.')

    args = parser.parse_args()

    # Train the model
    if args.run_type == 'terminal':
        train(args, flag='terminal')
    elif args.run_type == 'tuning':
        print('Tuning the parameters')
        tune_hyperparameters(args.run_type)