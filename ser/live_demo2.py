import json
import argparse
from os.path import isfile, isdir, join

import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import torch
import torch.nn.functional as F

from data_loader import calculate_spectrogram, CLASS_TO_ID, ID_TO_FULL_CLASS
from models import CNN, AttentionLSTM, FeatureEnsemble
from config import AcousticSpectrogramConfig, LinguisticConfig, EnsembleConfig
from text_preprocessing import Preprocessor, Word2VecWrapper

import sys
from os import listdir
from os.path import isfile

SAMPLE_RATE = 16000
DURATION_IN_SEC = 8
RECORDING_PATH = "/home/piosobc/Desktop/recordings"


def print_recordings(recordings):
    print("----------------------------------------------------------")
    print("Which recording do you want to test? Available recordings:")
    for id, name in recordings.items():
        print("[{}] {}".format(id, name))
    print("Press q to exit.")
    print("----------------------------------------------------------")


def print_probs(probabilities):
    # print("Output: {}".format(ID_TO_FULL_CLASS[np.argmax(probabilities)]))
    # print("Probabilities:")
    # for class_name, prob in zip(ID_TO_FULL_CLASS.values(), probabilities):
    #     print("{}: {}".format(class_name, round(float(prob),2)))
    #
    # print("\n")

    for prob_threshold in [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]:
        string = "{}| ".format(round(prob_threshold + 0.05, 2))
        for prob in probabilities:
            if prob > prob_threshold:
                string += "|X| "
            else:
                string += "    "
        print(string)
    # print("    ________________")
    print("    ----------------->")
    print("      N   H   S   A")


def classify(spectrogram, transcriptions_emb):
    with torch.no_grad():
        output = acoustic_model(spectrogram)
        probabilities_a = F.softmax(output, 1).squeeze(0).numpy()
        # print("Acoustic output: {}, probabilities: {}".format(ID_TO_FULL_CLASS[np.argmax(output.numpy())], np.round(probabilities_a, 2)))
        #
        output = linguistic_model(transcriptions_emb)
        probabilities_l = F.softmax(output, 1).squeeze(0).numpy()
        # print("Linguistic output: {}, probabilities: {}".format(ID_TO_FULL_CLASS[np.argmax(output.numpy())], np.round(probabilities_l, 2)))

        # predictions_acoustic = F.softmax(acoustic_model(spectrogram).squeeze(1), dim=1)
        # predictions_linguistic = F.softmax(linguistic_model(transcriptions_emb).squeeze(1), dim=1)
        alpha = 0.45
        probabilities = probabilities_a * alpha + probabilities_l * (1 - alpha)
        print("Ensemble output: \033[1m{}\033[0m".format(ID_TO_FULL_CLASS[np.argmax(probabilities)]))
        print("Probabilities: \033[1m{}\033[0m\n".format(np.round(probabilities, 2)))
        # output = ensemble_model(spectrogram, transcriptions_emb)
        # probabilities = F.softmax(output, 1).squeeze(0).numpy()
        print_probs(probabilities)



def get_spectrogram(file_path):
    spectrogram = calculate_spectrogram(file_path, False)
    spectrogram = np.expand_dims(spectrogram, axis=0)
    return spectrogram


def get_transcription(deepspeech_path, file_path):
    transcription = speech_to_text(join(deepspeech_path, "output_graph.pbmm"), join(deepspeech_path, "alphabet.txt"), join(deepspeech_path, "lm.binary"), join(deepspeech_path, "trie"), file_path)
    print("Automatic speech recognition: \033[1m{}\033[0m".format(transcription))
    preprocessed_transcription = Preprocessor.preprocess_one(transcription)
    transcriptions_emb = Word2VecWrapper.get_sentence_embedding(preprocessed_transcription, 30)
    transcriptions_emb = np.expand_dims(transcriptions_emb, axis=0)
    return transcriptions_emb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--linguistic_model", type=str, required=True)
    parser.add_argument("-a", "--acoustic_model", type=str, required=True)
    parser.add_argument("-e", "--ensemble_model", type=str, required=True)
    parser.add_argument("-d", "--deepspeech", type=str, required=True)
    parser.add_argument("-s", "--seconds", type=int, default=DURATION_IN_SEC)
    args = parser.parse_args()
    
    assert isfile(args.ensemble_model), "acoustic_model weights file does not exist"
    assert isfile(args.ensemble_model.replace(".torch", ".json")), "acoustic_model config file does not exist"
    assert isdir(args.deepspeech)
    
    from deepspeech_generator import speech_to_text

    """Converting model to specified hardware and format"""
    acoustic_cfg_json = json.load(open(args.acoustic_model.replace(".torch", ".json"), "r"))
    acoustic_cfg = AcousticSpectrogramConfig.from_json(acoustic_cfg_json)

    acoustic_model = CNN(acoustic_cfg)
    acoustic_model.float().to("cpu")

    try:
        acoustic_model.load_state_dict(torch.load(args.acoustic_model))
    except:
        print("Failed to load model from {} without device mapping. Trying to load with mapping to {}".format(
            args.acoustic_model, "cpu"))
        acoustic_model.load_state_dict(torch.load(args.acoustic_model, map_location="cpu"))

    acoustic_model.eval()

    linguistic_cfg_json = json.load(open(args.linguistic_model.replace(".torch", ".json"), "r"))
    linguistic_cfg = LinguisticConfig.from_json(linguistic_cfg_json)

    linguistic_model = AttentionLSTM(linguistic_cfg)
    linguistic_model.float().to("cpu")

    try:
        linguistic_model.load_state_dict(torch.load(args.linguistic_model))
    except:
        print("Failed to load model from {} without device mapping. Trying to load with mapping to {}".format(
            args.linguistic_model, "cpu"))
        linguistic_model.load_state_dict(torch.load(args.linguistic_model, map_location="cpu"))

    linguistic_model.eval()
    
    ensemble_cfg_json = json.load(open(args.ensemble_model.replace(".torch", ".json"), "r"))
    ensemble_cfg = EnsembleConfig.from_json(ensemble_cfg_json)
    ensemble_model = FeatureEnsemble(ensemble_cfg)
    ensemble_model.float().to("cpu")
    
    try:
        ensemble_model.load_state_dict(torch.load(args.ensemble_model))
    except:
        print("Failed to load model from {} without device mapping. Trying to load with mapping to {}".format(
            args.ensemble_model, "cpu"))
        ensemble_model.load_state_dict(torch.load(args.ensemble_model, map_location="cpu"))
    
    ensemble_model.eval()


def get_recordings(path):
    files = [file for file in listdir(path) if isfile(join(path, file))]
    files.sort()
    recordings = {str(i): recording for i, recording in enumerate(files)}
    return recordings


recordings = get_recordings(RECORDING_PATH)
speech_to_text(join(args.deepspeech, "output_graph.pbmm"), join(args.deepspeech, "alphabet.txt"), join(args.deepspeech, "lm.binary"), join(args.deepspeech, "trie"), join(RECORDING_PATH, recordings["0"]))
print_recordings(recordings)


for line in sys.stdin:
    recordings = get_recordings(RECORDING_PATH)
    choice = line.strip()
    if choice == "q":
        print("Exiting program")
        break
    elif choice not in recordings:
        print("Bad choice!")
        print_recordings(recordings)
        continue

    import subprocess
    import os

    FNULL = open(os.devnull, 'w')
    subprocess.call(["ffplay", "-nodisp", "-autoexit", join(RECORDING_PATH, recordings[choice])], stderr=FNULL)

    spectrogram = get_spectrogram(join(RECORDING_PATH, recordings[choice]))
    spectrogram = spectrogram
    transcriptions_emb = get_transcription(args.deepspeech, join(RECORDING_PATH, recordings[choice]))
    classify(spectrogram, transcriptions_emb)

    print_recordings(recordings)