import torch
import os
from time import gmtime, strftime
import json
import argparse

from models import AttentionLSTM as RNN, CNN
from model_utils import evaluate, train
from batch_iterator import BatchIterator
from data_loader import load_linguistic_dataset, load_acoustic_features_dataset, load_spectrogram_dataset
from utils import get_datetime, log, log_major, log_success, get_device, set_default_tensor
from config import LinguisticConfig, AcousticLLDConfig, AcousticSpectrogramConfig
from tensorboardX import SummaryWriter

MODEL_PATH = "saved_models"


def run_training(model, cfg, test_features, test_labels, train_data, train_labels, val_data, val_labels):
    tmp_run_path = "/tmp/model" + get_datetime()
    model_weights_path = "{}/{}".format(tmp_run_path, cfg.model_weights_name)
    model_config_path = "{}/{}".format(tmp_run_path, cfg.model_config_name)
    result_path = "{}/result.txt".format(tmp_run_path)
    os.makedirs(tmp_run_path, exist_ok=True)
    json.dump(cfg.to_json(), open(model_config_path, "w"))

    """Defining loss and optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(get_device())

    """Creating data generators"""
    test_iterator = BatchIterator(test_features, test_labels)
    train_iterator = BatchIterator(train_data, train_labels, cfg.batch_size)
    validation_iterator = BatchIterator(val_data, val_labels)

    train_loss = 999
    best_val_loss = 999
    train_acc = 0
    epochs_without_improvement = 0

    writer = SummaryWriter()

    """Running training"""
    for epoch in range(cfg.n_epochs):
        train_iterator.shuffle()
        if epochs_without_improvement == cfg.patience:
            break

        val_loss, val_cm = evaluate(model, validation_iterator, criterion)

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), model_weights_path)
            best_val_loss = val_loss
            best_val_acc = val_cm.accuracy
            best_val_unweighted_acc = val_cm.unweighted_accuracy
            epochs_without_improvement = 0
            log_success(" Epoch: {} | Val loss improved to {:.4f} | val acc: {:.3f} | weighted val acc: {:.3f} | train loss: {:.4f} | train acc: {:.3f} | saved model to {}.".format(
                epoch, best_val_loss, best_val_acc, best_val_unweighted_acc, train_loss, train_acc, model_weights_path
            ))

        train_loss, train_cm = train(model, train_iterator, optimizer, criterion, cfg.reg_ratio)
        train_acc = train_cm.accuracy

        writer.add_scalars('all/losses', {"val": val_loss, "train": train_loss}, epoch)
        writer.add_scalars('all/accuracy', {"val": val_cm.accuracy, "train": train_cm.accuracy}, epoch)
        writer.add_scalars('all/unweighted_acc', {"val": val_cm.unweighted_accuracy, "train": train_cm.unweighted_accuracy}, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/val_acc', val_cm.accuracy, epoch)
        writer.add_scalar('val/val_unweighted_acc', val_cm.unweighted_accuracy, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/train_acc', train_cm.accuracy, epoch)
        writer.add_scalar('train/train_unweighted_acc', train_cm.unweighted_accuracy, epoch)

        epochs_without_improvement += 1
    
        if not epoch % 1:
            log(f'| Epoch: {epoch+1} | Val Loss: {val_loss:.3f} | Val Acc: {val_cm.accuracy*100:.2f}% '
                f'| Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.3f}%', cfg.verbose)

    model.load_state_dict(torch.load(model_weights_path))
    test_loss, test_cm = evaluate(model, test_iterator, criterion)

    result = f'| Epoch: {epoch+1} | Test Loss: {test_loss:.3f} | Test Acc: {test_cm.accuracy*100:.2f}% | Weighted Test Acc: {test_cm.unweighted_accuracy*100:.2f}%\n Confusion matrix:\n {test_cm}'
    log_major("Train acc: {}".format(train_acc))
    log_major(result)
    log_major("Hyperparameters:{}".format(cfg.to_json()))
    with open(result_path, "w") as file:
        file.write(result)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

    output_path = "{}/{}_{:.3f}Acc_{:.3f}UAcc_{}".format(MODEL_PATH, cfg.model_name, test_cm.accuracy, test_cm.unweighted_accuracy, strftime("%Y-%m-%d_%H:%M:%S", gmtime()))
    os.rename(tmp_run_path, output_path)

    return test_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="linguistic")
    args = parser.parse_args()
    set_default_tensor()

    if args.model_type == "linguistic":
        cfg = LinguisticConfig()
        test_features, test_labels, val_features, val_labels, train_features, train_labels = load_linguistic_dataset()
        model = RNN(cfg)
    elif args.model_type == "acoustic-lld":
        cfg = AcousticLLDConfig()
        test_features, test_labels, val_features, val_labels, train_features, train_labels = load_acoustic_features_dataset()
        model = RNN(cfg)
    elif args.model_type == "acoustic-spectrogram":
        cfg = AcousticSpectrogramConfig()
        test_features, test_labels, val_features, val_labels, train_features, train_labels = load_spectrogram_dataset()
        model = CNN(cfg)
    else:
        raise Exception("model_type parameter has to be one of [linguistic|acoustic-lld|acoustic-spectrogram]")

    """Converting model to specified hardware and format"""
    model.float()
    model = model.to(get_device())

    print("Subsets sizes: test_features:{}, test_labels:{}, val_features:{}, val_labels:{}, train_features:{}, train_labels:{}".format(
        test_features.shape[0], test_labels.shape[0], val_features.shape[0], val_labels.shape[0], train_features.shape[0], train_labels.shape[0])
    )

    """Running training"""
    run_training(model, cfg, test_features, test_labels, train_features, train_labels, val_features, val_labels)
