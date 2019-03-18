import torch
import os
from time import gmtime, strftime
import json

from models import RNN
from model_ops import evaluate, train
from batch_iterator import BatchIterator
from data_loader import get_transcription_embeddings_and_labels
from utils import timeit, log, log_major, log_success


N_EPOCHS = 1000
PATIENCE = 50
REG_RATIO = 0.0015
BATCH_SIZE = 256
SEQ_LEN = 25
VERBOSE = True
LR = 0.001

MODEL_PATH = "saved_models"
TRANSCRIPTIONS_VAL_PATH = "data/iemocap_transcriptions_val.json"
TRANSCRIPTIONS_TRAIN_PATH = "data/iemocap_transcriptions_train.json"


@timeit
def run_training(**kwargs):
    model_run_path = MODEL_PATH + "/" + strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    model_weights_path = "{}/model.torch".format(model_run_path)
    model_config_path = "{}/config.json".format(model_run_path)
    result_path = "{}/result.txt".format(model_run_path)
    os.makedirs(model_run_path, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == "cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    log_major("Hyperparameters:{}".format(kwargs))
    json.dump(kwargs, open(model_config_path, "w"))

    batch_size = kwargs.pop("batch_size", BATCH_SIZE)
    seq_len = kwargs.pop("seq_len", SEQ_LEN)
    lr = kwargs.pop("lr", LR)
    reg_ratio = kwargs.pop("reg_ratio", REG_RATIO)

    model = RNN(**kwargs)
    model.float()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    best_valid_loss = 999
    epochs_without_improvement = 0

    """Creating data generators"""
    val_transcriptions, val_labels = get_transcription_embeddings_and_labels(TRANSCRIPTIONS_VAL_PATH, seq_len)
    train_transcriptions, train_labels = get_transcription_embeddings_and_labels(TRANSCRIPTIONS_TRAIN_PATH, seq_len)

    train_iterator = BatchIterator(train_transcriptions, train_labels, batch_size)
    validation_iterator = BatchIterator(val_transcriptions, val_labels, 100)

    train_loss=999
    train_acc=0

    """Running training"""
    for epoch in range(N_EPOCHS):
        if epochs_without_improvement == PATIENCE:
            break

        valid_loss, valid_acc, conf_mat = evaluate(model, validation_iterator, criterion)

        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), model_weights_path)
            best_valid_loss = valid_loss
            best_valid_acc = valid_acc
            best_conf_mat = conf_mat
            epochs_without_improvement = 0
            log_success(" Epoch: {} | Val loss improved to {:.4f} | val acc: {:.3f} | train loss: {:.4f} | train acc: {:.3f} | saved model to {}.".format(
                epoch, best_valid_loss, best_valid_acc, train_loss, train_acc, model_weights_path
            ))

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, reg_ratio)

        epochs_without_improvement += 1
    
        if not epoch % 1:
            log(f'| Epoch: {epoch+1} | Val Loss: {valid_loss:.3f} | Val Acc: {valid_acc*100:.2f}% '
                f'| Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.3f}%', VERBOSE)

    result = f'| Epoch: {epoch+1} | Val Loss: {best_valid_loss:.3f} | Val Acc: {best_valid_acc*100:.2f}% | ' \
             f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.3f}% \n Confusion matrix:\n {best_conf_mat}'
    log_major(result)
    log_major("Hyperparameters:{}".format(kwargs))
    with open(result_path, "w") as file:
        file.write(result)


    # model.load_state_dict(torch.load(model_weights_path))
    # test_loss, test_acc = evaluate(model, test_iterator, criterion)
    # print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')


if __name__ == "__main__":
    run_training()
