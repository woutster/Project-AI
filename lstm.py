from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

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

    # Initialise model
    model = LSTM(batch_size=args.batch_size,
                input_size=args.input_size,
                lstm_num_hidden=args.lstm_num_hidden,
                lstm_num_layers=args.lstm_num_layers,
                device=device
    )
    model.to(device)

    # Set up the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)

    # TODO: data preproccessing




    # Iterate over data
    for step, datapoint in enumerate(data):

        # TODO: reshape data?
        x = datapoint

        optimizer.zero_grad()
        out, (h,c) = model(x)

        # TODO: not reshape data?
        loss = criterion(out.transpose(2,1), y)
        loss.backward()
        optimizer.step()

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
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(1e6), help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    args = parser.parse_args()

    # Train the model
    train(args)
