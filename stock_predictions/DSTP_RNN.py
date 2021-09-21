"""
References:
- https://github.com/Zhenye-Na/DA-RNN
- https://github.com/arleigh418/Paper-Implementation-DSTP-RNN-For-Stock-Prediction-Based-On-DA-RNN
"""

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable

# torch.cpu.is_available()

from stock_predictions import ROOT

matplotlib.use('Agg')


def count_values(truth, pred):
    count_avg = 0
    assert len(truth) == len(pred)
    for x in range(len(truth)):
        count_avg += abs(truth[x] - pred[x])
    return count_avg / len(truth)


def read_data(input_path):
    """Read nasdaq stocks data.
    Args:
        input_path (str): directory to nasdaq dataset.
    Returns:
        X (np.ndarray): features.
        y (np.ndarray): ground truth.
    """
    df = pd.read_csv(input_path)
    df_no_date = df.drop(["Close", 'Date'], axis=1)
    X = df_no_date.values
    y = df['Close'].values
    return X, y


class Encoder(nn.Module):
    def __init__(self, T, input_size, encoder_num_hidden, parallel=False):
        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.parallel = parallel
        self.T = T

        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.encoder_lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.encoder_num_hidden)
        self.encoder_lstm2 = nn.LSTM(
            input_size=self.input_size, hidden_size=self.encoder_num_hidden
        )
        # Construct Input Attention Mechanism via deterministic attention model
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        self.encoder_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + self.T - 1, out_features=1, bias=True
        )  # 1033
        # W_s[h_{t-1} ; s_{t-1}] + U_s[x^k ; y^k]
        self.encoder_attn2 = nn.Linear(
            in_features=2 * self.encoder_num_hidden + 2 * self.T - 2, out_features=1, bias=True
        )

    def forward(self, X, y_prev):
        """forward.

        Args:
            X

        """
        X_tilde = Variable(X.data.new(X.size(0), self.T - 1, self.input_size).zero_())
        X_encoded = Variable(X.data.new(X.size(0), self.T - 1, self.encoder_num_hidden).zero_())

        X_tilde2 = Variable(X.data.new(X.size(0), self.T - 1, self.input_size).zero_())
        X_encoded2 = Variable(X.data.new(X.size(0), self.T - 1, self.encoder_num_hidden).zero_())

        # hidden, cell: initial states with dimension hidden_size

        h_n = self._init_states(X)
        s_n = self._init_states(X)

        hs_n = self._init_states(X)
        ss_n = self._init_states(X)
        # y_prev = y_prev.view()

        y_prev = y_prev.view(len(X), self.T - 1, 1)

        # print(h_n.size())  # 1 233 512
        # print(s_n.size())

        for t in range(self.T - 1):
            # Phase one attention
            # batch_size * input_size * (2*hidden_size + T - 1)
            x = torch.cat(
                (
                    h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),  # 233 363 1033
                    s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                    X.permute(0, 2, 1),
                ),
                dim=2,
            )

            # test = x.view(-1, self.encoder_num_hidden * 2 + self.T - 1)
            # print(test.size()) #84579 1033

            x = self.encoder_attn(x.view(-1, self.encoder_num_hidden * 2 + self.T - 1))  # 84579 1

            # get weights by softmax
            alpha = F.softmax(x.view(-1, self.input_size))  # 233x363

            # get new input for LSTM
            x_tilde = torch.mul(alpha, X[:, t, :])  # 233x363
            # print(x_tilde.size())

            # encoder LSTM
            self.encoder_lstm.flatten_parameters()
            _, final_state = self.encoder_lstm(x_tilde.unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]

            # Phase two attention from DSTP-RNN Paper

            x2 = torch.cat(
                (
                    hs_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),  # 233 363 1042
                    ss_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                    X.permute(0, 2, 1),
                    y_prev.repeat(1, 1, self.input_size).permute(0, 2, 1),
                ),
                dim=2,
            )

            x2 = self.encoder_attn2(x2.view(-1, self.encoder_num_hidden * 2 + 2 * self.T - 2))

            alpha2 = F.softmax(x2.view(-1, self.input_size))  # 233x363

            x_tilde2 = torch.mul(alpha2, x_tilde)

            self.encoder_lstm2.flatten_parameters()
            _, final_state2 = self.encoder_lstm2(x_tilde2.unsqueeze(0), (hs_n, ss_n))
            hs_n = final_state2[0]
            ss_n = final_state2[1]
            # print(x_tilde2.size())
            X_tilde2[:, t, :] = x_tilde2
            X_encoded2[:, t, :] = hs_n

        return X_tilde2, X_encoded2

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X
        Returns:
            initial_hidden_states

        """
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        initial_states = Variable(X.data.new(1, X.size(0), self.encoder_num_hidden).zero_())
        return initial_states


class Decoder(nn.Module):
    def __init__(self, T, decoder_num_hidden, encoder_num_hidden):
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T

        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_num_hidden + encoder_num_hidden, encoder_num_hidden),
            nn.Tanh(),
            nn.Linear(encoder_num_hidden, 1),
        )
        self.lstm_layer = nn.LSTM(input_size=1, hidden_size=decoder_num_hidden)
        self.fc = nn.Linear(encoder_num_hidden + 1, 1)
        self.fc_final_price = nn.Linear(decoder_num_hidden + encoder_num_hidden, 1)
        self.fc.weight.data.normal_()

    def forward(self, X_encoed, y_prev):
        """forward."""
        d_n = self._init_states(X_encoed)
        c_n = self._init_states(X_encoed)

        for t in range(self.T - 1):
            x = torch.cat(
                (
                    d_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                    c_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                    X_encoed,
                ),
                dim=2,
            )
            beta = F.softmax(
                self.attn_layer(
                    x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)
                ).view(-1, self.T - 1)
            )
            # Eqn. 14: compute context vector
            # batch_size * encoder_hidden_size
            context = torch.bmm(beta.unsqueeze(1), X_encoed)[:, 0, :]
            if t < self.T - 1:
                # Eqn. 15
                # batch_size * 1
                y_tilde = self.fc(torch.cat((context, y_prev[:, t].unsqueeze(1)), dim=1))
                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, final_states = self.lstm_layer(y_tilde.unsqueeze(0), (d_n, c_n))
                # 1 * batch_size * decoder_num_hidden
                d_n = final_states[0]
                # 1 * batch_size * decoder_num_hidden
                c_n = final_states[1]
        # Eqn. 22: final output
        final_temp_y = torch.cat((d_n[0], context), dim=1)
        y_pred_price = self.fc_final_price(final_temp_y)

        return y_pred_price

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X
        Returns:
            initial_hidden_states

        """
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        initial_states = X.data.new(1, X.size(0), self.decoder_num_hidden).zero_()
        return initial_states


class DSTP_rnn(nn.Module):
    def __init__(
        self,
        X,
        y,
        T,
        encoder_num_hidden,
        decoder_num_hidden,
        batch_size,
        learning_rate,
        epochs,
        parallel=False,
    ):

        super(DSTP_rnn, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.parallel = parallel
        self.shuffle = False
        self.epochs = epochs
        self.T = T
        self.X = X
        self.y = y

        self.Encoder = Encoder(input_size=X.shape[1], encoder_num_hidden=encoder_num_hidden, T=T)
        self.Decoder = Decoder(
            encoder_num_hidden=encoder_num_hidden, decoder_num_hidden=decoder_num_hidden, T=T
        )
        self.Encoder = self.Encoder.cpu()
        self.Decoder = self.Decoder.cpu()
        # Loss function
        self.criterion_price = nn.MSELoss()

        if self.parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad, self.Encoder.parameters()),
            lr=self.learning_rate,
        )
        self.decoder_optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad, self.Decoder.parameters()),
            lr=self.learning_rate,
        )

        # Training set
        self.train_timesteps = int(self.X.shape[0] * 0.8)
        self.input_size = self.X.shape[1]

    def train(self):

        """training process."""
        iter_per_epoch = int(np.ceil(self.train_timesteps * 1.0 / self.batch_size))
        self.iter_losses = np.zeros(self.epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(self.epochs)
        n_iter = 0

        val_record = 0

        for epoch in range(self.epochs):
            if self.shuffle:
                ref_idx = np.random.permutation(self.train_timesteps - self.T)
            else:
                ref_idx = np.array(range(self.train_timesteps - self.T))
            idx = 0

            while idx < self.train_timesteps:
                # get the indices of X_train
                indices = ref_idx[idx : (idx + self.batch_size)]
                # x = np.zeros((self.T - 1, len(indices), self.input_size))
                x = np.zeros((len(indices), self.T - 1, self.input_size))
                y_prev = np.zeros((len(indices), self.T - 1))
                y_gt = self.y[indices + self.T]

                # format x into 3D tensor
                for bs in range(len(indices)):
                    x[bs, :, :] = self.X[indices[bs] : (indices[bs] + self.T - 1), :]
                    y_prev[bs, :] = self.y[indices[bs] : (indices[bs] + self.T - 1)]

                loss = self.train_forward(x, y_prev, y_gt)

                self.iter_losses[epoch * iter_per_epoch + idx // self.batch_size] = loss

                idx += self.batch_size
                n_iter += 1

                if n_iter % 10000 == 0 and n_iter != 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                self.epoch_losses[epoch] = np.mean(
                    self.iter_losses[range(epoch * iter_per_epoch, (epoch + 1) * iter_per_epoch)]
                )

            if epoch % 10 == 0:
                print(
                    "Epochs: ", epoch, " Iterations: ", n_iter, " Loss: ", self.epoch_losses[epoch]
                )
            if epoch % 1000 == 0 and epoch != 0:
                torch.save(model.state_dict(), f'{ROOT}/models/dstprnn_model_{epoch}.pkl')

    def train_forward(self, X, y_prev, y_gt):
        # zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.Encoder(
            Variable(torch.from_numpy(X).type(torch.FloatTensor).cpu()),
            Variable(torch.from_numpy(y_prev).type(torch.FloatTensor).cpu()),
        )  # cpu
        y_pred_price = self.Decoder(
            input_encoded, Variable(torch.from_numpy(y_prev).type(torch.FloatTensor)).cpu()
        )  # cpu

        y_true_price = torch.from_numpy(y_gt).type(torch.FloatTensor)
        y_true_price = y_true_price.view(-1, 1).cpu()  # cpu
        loss = self.criterion_price(y_pred_price, y_true_price)
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return loss.item()

    def val(self):
        """validation."""
        pass

    def test(self, on_train=False):
        """test."""

        if on_train:
            y_pred_price = np.zeros(self.train_timesteps - self.T + 1)
        else:
            y_pred_price = np.zeros(self.X.shape[0] - self.train_timesteps)
        i = 0
        while i < len(y_pred_price):
            batch_idx = np.array(range(len(y_pred_price)))[i : (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T - 1))

            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(batch_idx[j], batch_idx[j] + self.T - 1), :]
                    y_history[j, :] = self.y[range(batch_idx[j], batch_idx[j] + self.T - 1)]
                else:
                    X[j, :, :] = self.X[
                        range(
                            batch_idx[j] + self.train_timesteps - self.T,
                            batch_idx[j] + self.train_timesteps - 1,
                        ),
                        :,
                    ]
                    y_history[j, :] = self.y[
                        range(
                            batch_idx[j] + self.train_timesteps - self.T,
                            batch_idx[j] + self.train_timesteps - 1,
                        )
                    ]

            y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).cpu())
            _, input_encoded = self.Encoder(
                Variable(torch.from_numpy(X).type(torch.FloatTensor).cpu()),
                Variable(y_history).cpu(),
            )  # cpu

            y_pred_price = self.Decoder(input_encoded, y_history)

            y_pred_price = y_pred_price[i : (i + self.batch_size)]
            y_pred_price = y_pred_price.cpu().detach().numpy()[:, 0]

            i += self.batch_size
        return y_pred_price


if __name__ == '__main__':
    X, y = read_data(f"{ROOT}/data/2324.TW.csv")
    model = DSTP_rnn(X, y, 10, 128, 128, 128, 0.001, epochs=100)
    # model = DSTP_rnn(X, y, 10, 128, 128, 128, 0.001, epochs=7000)
    model.train()
    torch.save(model.state_dict(), f=f'{ROOT}/models/dstprnn_model.pkl')
    # model = torch.load(f'{ROOT}/models/dstprnn_model.pkl')
    pred = model.test()
    print(pred)
