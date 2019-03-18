import torch


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
        x = torch.cuda.FloatTensor(x)
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
