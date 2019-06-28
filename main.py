import pandas as pd
import numpy as np
import argparse

import get_gps
import process_speed
import lstm


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--lstm_num_hidden', type=int, default=1, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=1, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
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

    # Reads the initial gps_filter file and creates a geofence
    get_gps.run_get_gps()

    # Combines weather data with geofenced file, also calculates cmf
    process_speed.run_process_speed()

    # Train the LSTM model
    lstm.train(args)
