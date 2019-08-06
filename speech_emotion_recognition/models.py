import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from speech_emotion_recognition.utils import get_device


class LoadableModule(torch.nn.Module):
    def load(self, model_path):
        try:
            super(LoadableModule, self).load_state_dict()
        except:
            print("Failed to load model from {} without device mapping. Trying to load with mapping to {}".format(
                model_path, get_device()))
            super(LoadableModule, self).load_state_dict(torch.load(model_path, map_location=get_device()))

    def forward(self, input):
        raise Exception("Not implemented!")


class AttentionLSTM(LoadableModule):
    """Taken from https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM_Attn.py"""
    def __init__(self, cfg):
        """
        LSTM with self-Attention model.
        :param cfg: Linguistic config object
        """
        super(AttentionLSTM, self).__init__()
        self.batch_size = cfg.batch_size
        self.output_size = cfg.num_classes
        self.hidden_size = cfg.hidden_dim
        self.embedding_length = cfg.emb_dim

        self.dropout = torch.nn.Dropout(cfg.dropout)
        self.dropout2 = torch.nn.Dropout(cfg.dropout2)

        self.lstm = nn.LSTM(self.embedding_length, self.hidden_size)
        self.label = nn.Linear(self.hidden_size, self.output_size)

    def attention_net(self, lstm_output, final_state):
        """
        This method computes soft alignment scores for each of the hidden_states and the last hidden_state of the LSTM.
        Tensor Sizes :
            hidden.shape = (batch_size, hidden_size)
            attn_weights.shape = (batch_size, num_seq)
            soft_attn_weights.shape = (batch_size, num_seq)
            new_hidden_state.shape = (batch_size, hidden_size)

        :param lstm_output: Final output of the LSTM which contains hidden layer outputs for each sequence.
        :param final_state: Final time-step hidden state (h_n) of the LSTM
        :return: Context vector produced by performing weighted sum of all hidden states with attention weights
        """

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def extract(self, input):
        input = input.transpose(0, 1)
        input = self.dropout(input)

        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        output = output.permute(1, 0, 2)

        attn_output = self.attention_net(output, final_hidden_state)
        return attn_output

    def classify(self, attn_output):
        attn_output = self.dropout2(attn_output)
        logits = self.label(attn_output)
        return logits.squeeze(1)

    def forward(self, input):
        attn_output = self.extract(input)
        logits = self.classify(attn_output)
        return logits


class CNN(LoadableModule):
    def __init__(self, cfg):
        super(CNN, self).__init__()
        self.conv_layers = self._build_conv_layers(cfg)
        self.out_size = cfg.input_size / (cfg.pool_size**len(cfg.num_filters))
        self.flat_size = cfg.num_filters[len(cfg.num_filters)-1] * self.out_size**2
        self.fc2 = nn.Linear(self.flat_size, cfg.num_classes)
        self.dropout = torch.nn.Dropout(cfg.dropout)

    def _build_conv_layers(self, cfg):
        conv_layers = []
        num_channels = [1] + cfg.num_filters
        for i in range(len(num_channels)-1):
            conv_layers.append(nn.Conv2d(num_channels[i], num_channels[i+1], cfg.conv_size, padding=1))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(cfg.pool_size, cfg.pool_size))
        return nn.Sequential(*conv_layers)

    def extract(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(-1, self.flat_size)
        return x

    def classify(self, x):
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.extract(x)
        x = self.classify(x)
        return x


class FeatureEnsemble(LoadableModule):
    def __init__(self, cfg, acoustic_model=None, linguistic_model=None):
        super(FeatureEnsemble, self).__init__()
        self.acoustic_model = acoustic_model if acoustic_model is not None else CNN(cfg.acoustic_config)
        self.linguistic_model = linguistic_model if linguistic_model is not None else AttentionLSTM(cfg.linguistic_config)
        self.feature_size = self.linguistic_model.hidden_size + self.acoustic_model.flat_size
        self.fc = nn.Linear(self.feature_size, 4)
        self.dropout = torch.nn.Dropout(0.7)

    def forward(self, input_tuple):
        acoustic_features, linguistic_features = input_tuple
        acoustic_output_features = self.acoustic_model.extract(acoustic_features)
        linguistic_output_features = self.linguistic_model.extract(linguistic_features)
        all_features = torch.cat((acoustic_output_features, linguistic_output_features), 1)
        return self.fc(self.dropout(all_features))

    @property
    def name(self):
        return "Feature Ensemble"


class DecisionEnsemble:
    def __init__(self, acoustic_model, linguistic_model):
        self.acoustic_model = acoustic_model
        self.linguistic_model = linguistic_model

    def __call__(self, input_tuple):
        acoustic_input, linguistic_input = input_tuple
        acoustic_output = F.log_softmax(self.acoustic_model(acoustic_input).squeeze(1), dim=1)
        linguistic_output = F.log_softmax(self.linguistic_model(linguistic_input).squeeze(1), dim=1)
        return self._ensemble_function(acoustic_output, linguistic_output)

    def eval(self):
        self.linguistic_model.eval()
        self.acoustic_model.eval()

    def _ensemble_function(self, acoustic_input, linguistic_input):
        raise Exception("Not Implemented!")

    @property
    def name(self):
        raise Exception("Not Implemented!")


class AverageEnsemble(DecisionEnsemble):
    def __init__(self, acoustic_model, linguistic_model):
        super(AverageEnsemble, self).__init__(acoustic_model, linguistic_model)

    def _ensemble_function(self, acoustic_output, linguistic_output):
        return (acoustic_output + linguistic_output)/2

    @property
    def name(self):
        return "Average Ensemble"


class WeightedAverageEnsemble(DecisionEnsemble):
    def __init__(self, acoustic_model, linguistic_model, alpha):
        super(WeightedAverageEnsemble, self).__init__(acoustic_model, linguistic_model)
        self.alpha = alpha

    def _ensemble_function(self, acoustic_output, linguistic_output):
        return acoustic_output * self.alpha + linguistic_output * (1 - self.alpha)

    @property
    def name(self):
        return "Weighted Average Ensemble"


class ConfidenceEnsemble(DecisionEnsemble):
    def __init__(self, acoustic_model, linguistic_model):
        super(ConfidenceEnsemble, self).__init__(acoustic_model, linguistic_model)

    def _ensemble_function(self, acoustic_output, linguistic_output):
        predictions = np.zeros(acoustic_output.shape)
        for i in range(acoustic_output.shape[0]):
            predictions[i] = acoustic_output[i] if acoustic_output[i].max() > linguistic_output[i].max() else linguistic_output[i]
        return torch.Tensor(predictions)

    @property
    def name(self):
        return "Confidence Ensemble"


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    # cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    num_filters = 16
    cfg = [(num_filters, 2), (num_filters, 2)]

    def __init__(self, cfg, num_classes=4):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layers = self._make_layers(in_planes=16)
        self.linear = nn.Linear(256, num_classes)
        self.dropout = torch.nn.Dropout(0.5)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.Tensor(x)
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out