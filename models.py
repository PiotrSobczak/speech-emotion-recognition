import torch
import torch.nn as nn
from torch.nn import functional as F


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

        return output_logits


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


class AttentionModel(torch.nn.Module):
    """Taken from https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM_Attn.py"""
    def __init__(self, cfg):
        super(AttentionModel, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        --------

        """        
        self.batch_size = cfg.batch_size
        self.output_size = cfg.num_classes
        self.hidden_size = cfg.hidden_dim
        self.embedding_length = cfg.emb_dim

        self.dropout = torch.nn.Dropout(cfg.dropout)
        self.dropout2 = torch.nn.Dropout(cfg.dropout2)
        self.lstm = nn.LSTM(self.embedding_length, self.hidden_size)
        self.label = nn.Linear(self.hidden_size, self.output_size)

    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output, final_state):

        """
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)

        """

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, input, batch_size=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
        final_output.shape = (batch_size, output_size)

        """
        # TODO CHECK THIS h_0 and c_0 reset
        # if batch_size is None:
        #     h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
        #     c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
        # else:
        #     h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        #     c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())

        input = torch.Tensor(input)
        input = self.dropout(input)
        output, (final_hidden_state, final_cell_state) = self.lstm(input)#, ( h_0, c_0))  # final_hidden_state.size() = (1, batch_size, hidden_size)
        output = output.permute(1, 0, 2)  # output.size() = (batch_size, num_seq, hidden_size)

        attn_output = self.attention_net(output, final_hidden_state)
        attn_output = self.dropout2(attn_output)
        logits = self.label(attn_output)

        return logits
