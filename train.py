import torch.nn as nn
import torch
import torch.optim as optim
from batch_iterator import BatchIterator
from data_loader import get_transcription_embeddings_and_labels
import torch.tensor
import json
import os
from time import gmtime, strftime

EMBEDDING_DIM = 400
HIDDEN_DIM = 800
OUTPUT_DIM = 1
DROPOUT = 0.0
N_EPOCHS = 100
PATIENCE = 3
REG_RATIO=0.00001
NUM_LAYERS=1
BIDIRECTIONAL = False
VERBOSE=False
MODEL_PATH = "models"
MODEL_RUN_PATH = MODEL_PATH + "/" + strftime("%Y-%m-%d_%H:%M:%S", gmtime())
MODEL_CONFIG = "{}/model.config".format(MODEL_RUN_PATH)
MODEL_WEIGHTS = "{}/model.torch".format(MODEL_RUN_PATH)

TRANSCRIPTIONS_VAL_PATH = "data/iemocap_transcriptions_val.json"
TRANSCRIPTIONS_TRAIN_PATH = "data/iemocap_transcriptions_train.json"
NUM_CLASSES = 4

class RNN(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS, config=None, model_config=MODEL_CONFIG, reg_ratio=REG_RATIO):
        super().__init__()
        log("---------------- NUM_LAYERS={}, HIDDEN_DIM={}, DROPOUT={}, REG_RATIO={}, BIDIR={}----------------".format(num_layers, hidden_dim, dropout, reg_ratio, BIDIRECTIONAL))


        if config is not None:
            embedding_dim = int(config["embedding_dim"])
            hidden_dim = int(config["hidden_dim"])
            dropout = float(config["dropout"])
        else:
            json.dump({"embedding_dim": embedding_dim, "hidden_dim": hidden_dim, "dropout": dropout, "reg_ratio": REG_RATIO, "n_layers": NUM_LAYERS},
open(model_config,
"w"))

        fc_size = hidden_dim * 2 if BIDIRECTIONAL else hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=NUM_LAYERS, bidirectional=BIDIRECTIONAL, dropout=dropout)
        self.fc = nn.Linear(fc_size, NUM_CLASSES)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """ x = [sent len, batch size, emb dim] """
        x = torch.cuda.FloatTensor(x)
        x = self.dropout(x)

        """ output = [sent len, batch size, hid dim * num directions]
            hidden&cell = [num layers * num directions, batch size, hid dim] """
        output, (hidden, cell) = self.lstm(x)

        """ concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers and apply dropout """
        if BIDIRECTIONAL:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        hidden = hidden.squeeze(0).float()
        return self.fc(hidden)


def accuracy(preds, y):
    """ Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8 """
    rounded_preds = torch.argmax(preds,1)

    correct = (rounded_preds == y).float()
    acc = correct.sum()/len(correct)
    return acc


def train(model, iterator, optimizer, criterion, reg_ratio=REG_RATIO):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch, labels in iterator():
        optimizer.zero_grad()

        predictions = model(batch).squeeze(1)

        loss = criterion(predictions, labels)
        acc = accuracy(predictions, labels)

        reg_loss = 0
        for param in model.parameters():
            reg_loss += param.norm(2)

        total_loss = loss + reg_ratio*reg_loss
        total_loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch, labels in iterator():
            predictions = model(batch).squeeze(1)

            loss = criterion(predictions.float(), labels)

            acc = accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


#@timeit
def run_training(**kwargs):
    model_weights = "{}/model.torch".format(kwargs.get("model_run_path", MODEL_RUN_PATH))
    os.makedirs(MODEL_RUN_PATH, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == "cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    model = RNN(**kwargs)
    model.float()
    model = model.to(device)
    model.float()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    best_valid_loss = 999
    epochs_without_improvement = 0

    """Creating data generators"""
    val_trainscriptions, val_labels = get_transcription_embeddings_and_labels(TRANSCRIPTIONS_VAL_PATH)
    train_trainscriptions, train_labels = get_transcription_embeddings_and_labels(TRANSCRIPTIONS_TRAIN_PATH)

    train_iterator = BatchIterator(train_trainscriptions, train_labels)
    validation_iterator = BatchIterator(val_trainscriptions, val_labels)

    for epoch in range(N_EPOCHS):
        if epochs_without_improvement == PATIENCE:
            break

        valid_loss, valid_acc = evaluate(model, validation_iterator, criterion)
        log(f'| Epoch: {epoch:02} | Val Loss: {valid_loss:.4f} | Val Acc: {valid_acc*100:.3f}%')
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), model_weights)
            print("Val loss improved from {} to {}. Saving model to {}.".format(best_valid_loss, valid_loss,
                                                                                model_weights))
            best_valid_loss = valid_loss
            epochs_without_improvement = 0

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, kwargs["reg_ratio"])
        log(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.3f}%')

        epochs_without_improvement += 1
    
        #if not epoch % 1:
        print(f'| Val Loss: {valid_loss:.3f} | Val Acc: {valid_acc*100:.2f}% '
              f'| Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.3f}%')
    # model.load_state_dict(torch.load(model_weights))
    # test_loss, test_acc = evaluate(model, test_iterator, criterion)
    # print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')


def log(log_message):
    if VERBOSE:
        print(log_message)


if __name__ == "__main__":
    import numpy as np
    params = {}
    params["num_layers"] = np.random.randint(1, 4)
    params["hidden_dim"] = np.random.randint(64, 1200)
    params["dropout"] = 0.1+np.random.rand()*0.85
    params["reg_ratio"] = np.random.rand()*0.00001
    run_training(**params)
