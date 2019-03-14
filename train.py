import torch
import os
from time import gmtime, strftime

from models import RNN
from model_ops import evaluate, train
from batch_iterator import BatchIterator
from data_loader import get_transcription_embeddings_and_labels
from utils import timeit, log


N_EPOCHS = 1000
PATIENCE = 300
REG_RATIO = 0.00001

VERBOSE = True

MODEL_PATH = "saved_models"
MODEL_RUN_PATH = MODEL_PATH + "/" + strftime("%Y-%m-%d_%H:%M:%S", gmtime())
TRANSCRIPTIONS_VAL_PATH = "data/iemocap_transcriptions_val.json"
TRANSCRIPTIONS_TRAIN_PATH = "data/iemocap_transcriptions_train.json"


@timeit
def run_training(**kwargs):
    model_weights = "{}/model.torch".format(kwargs.get("model_run_path", MODEL_RUN_PATH))
    os.makedirs(MODEL_RUN_PATH, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == "cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    kwargs["model_config_path"] = "{}/model.config".format(MODEL_RUN_PATH)
    kwargs["reg_ratio"] = kwargs.get("reg_ratio", REG_RATIO)
    model = RNN(**kwargs)
    model.float()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    best_valid_loss = 999
    epochs_without_improvement = 0

    """Creating data generators"""
    val_transcriptions, val_labels = get_transcription_embeddings_and_labels(TRANSCRIPTIONS_VAL_PATH)
    train_transcriptions, train_labels = get_transcription_embeddings_and_labels(TRANSCRIPTIONS_TRAIN_PATH)

    train_iterator = BatchIterator(train_transcriptions, train_labels)
    validation_iterator = BatchIterator(val_transcriptions, val_labels)

    """Running training"""
    for epoch in range(N_EPOCHS):
        if epochs_without_improvement == PATIENCE:
            break

        valid_loss, valid_acc, conf_mat = evaluate(model, validation_iterator, criterion)

        log(f'| Epoch: {epoch:02} | Val Loss: {valid_loss:.4f} | Val Acc: {valid_acc*100:.3f}%', VERBOSE)
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), model_weights)
            print("Val loss improved to {}. Saved model to {}.".format(best_valid_loss, valid_loss, model_weights))
            best_valid_loss = valid_loss
            epochs_without_improvement = 0

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, kwargs["reg_ratio"])
        # log(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.3f}%', VERBOSE)

        epochs_without_improvement += 1
    
        if not epoch % 1:
            log(f'| Epoch: {epoch+1} | Val Loss: {valid_loss:.3f} | Val Acc: {valid_acc*100:.2f}% '
                f'| Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.3f}%', VERBOSE)
            log(conf_mat, VERBOSE)
    # model.load_state_dict(torch.load(model_weights))
    # test_loss, test_acc = evaluate(model, test_iterator, criterion)
    # print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')


if __name__ == "__main__":
    run_training()