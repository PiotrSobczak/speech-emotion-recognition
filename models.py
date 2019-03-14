import torch


NUM_CLASSES = 4
EMB_DIM = 400
HIDDEN_DIM = 800
OUTPUT_DIM = 1
DROPOUT = 0.5
N_LAYERS = 1
BIDIRECTIONAL = False
VERBOSE = True


class RNN(torch.nn.Module):
    def __init__(self, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, n_layers=N_LAYERS, bidirectional=BIDIRECTIONAL):
        super().__init__()
        fc_size = hidden_dim * 2 if BIDIRECTIONAL else hidden_dim
        self.lstm = torch.nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)

        for layer_p in self.lstm._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    torch.nn.init.xavier_uniform_(self.lstm.__getattr__(p))
        self.fc = torch.nn.Linear(fc_size, NUM_CLASSES)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        """ x = [sent len, batch size, emb dim] """
        x = torch.cuda.FloatTensor(x)
        x = self.dropout(x)

        """ output      = [sent len, batch size, hid dim * num directions]
            hidden&cell = [num layers * num directions, batch size, hid dim] """
        output, (hidden, cell) = self.lstm(x)

        """ concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers and apply dropout """
        if BIDIRECTIONAL:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        hidden = hidden.squeeze(0).float()
        return self.fc(hidden)
