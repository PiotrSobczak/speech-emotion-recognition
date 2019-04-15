import torch
import torch.nn as nn
import torch.nn.functional as F


def init_linear(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal(m.weight)
        nn.init.constant(m.bias, 0)


class AttnDecoderRNN(nn.Module):
    """ Based on: https://medium.com/apache-mxnet/sentiment-analysis-via-self-attention-with-mxnet-gluon-dc774d38ba69 
    Also take a look at : https://github.com/gogyzzz/localatt_emorecog/blob/master/localatt/localatt.py"""
    def __init__(self, cfg):
        super(AttnDecoderRNN, self).__init__()
        self.do2 = nn.Dropout()

        self.blstm = torch.nn.LSTM(cfg.emb_dim, cfg.hidden_dim, num_layers=cfg.n_layers, bidirectional=cfg.bidirectional)

        self.d_dim = 64
        self.r_dim = 32

        self.fc1 = nn.Linear(cfg.hidden_dim, self.d_dim)
        self.fc2 = nn.Linear(self.d_dim, self.r_dim)
        self.fc3 = nn.Linear(self.r_dim, cfg.num_classes)

        self.apply(init_linear)

    def forward(self, input):
        """ input = [seq_len, batch size, emb_dim] """
        input = torch.Tensor(input)

        """ output = [seq_len, batch size, hidden_dim] """
        output, (hidden, cell) = self.blstm(input)

        """ output_trans = [seq_len, batch size, d_dim] """
        output_trans = self.fc1(output)
        output_trans = F.tanh(output_trans)

        """ output_scores = [seq_len, batch size, r_dim] """
        output_scores = self.fc2(output_trans)
        attention = F.softmax(output_scores)

        """ output_trans2_sum = [batch size, r_dim] """
        outputs_scaled = torch.sum(attention, dim=0)

        """ output_logits = [batch size, num_classes] """
        output_logits = self.fc3(outputs_scaled)

        return F.softmax(output_logits)


class RNN(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        fc_size = cfg.hidden_dim * 2 if cfg.bidirectional else cfg.hidden_dim
        self.lstm = torch.nn.LSTM(cfg.emb_dim, cfg.hidden_dim, num_layers=cfg.n_layers, bidirectional=cfg.bidirectional)

        self.fc = torch.nn.Linear(fc_size, cfg.num_classes)

        self.dropout = torch.nn.Dropout(cfg.dropout)
        self.dropout2 = torch.nn.Dropout(cfg.dropout2)

        """ Xavier Initialization """
        for layer_p in self.lstm._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    torch.nn.init.xavier_uniform_(self.lstm.__getattr__(p))

        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        """ x = [sent len, batch size, emb dim] """
        x = torch.Tensor(x)
        x = self.dropout(x)

        """ output      = [sent len, batch size, hid dim * num directions]
            hidden&cell = [num layers * num directions, batch size, hid dim] """
        output, (hidden, cell) = self.lstm(x)

        """ concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers and apply dropout """
        if self.lstm.bidirectional:
            hidden = self.dropout2(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout2(hidden[-1, :, :])
        hidden = hidden.squeeze(0).float()
        return self.fc(hidden)
