import json
import argparse
from os.path import isfile, isdir, join

import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import torch
import torch.nn.functional as F

from data_loader import calculate_spectrogram, CLASS_TO_ID, ID_TO_CLASS
from models import CNN, AttentionModel, EnsembleModel
from config import AcousticSpectrogramConfig, LinguisticConfig, EnsembleConfig
from preprocessing import Preprocessor, Word2VecWrapper

SAMPLE_RATE = 16000
DURATION_IN_SEC = 8
TMP_FILENAME = 'output.wav'

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
    
    print("Starting recording for {} seconds...".format(args.seconds))
    myrecording = sd.rec(int(args.seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    print("Recording finished")
    
    write(TMP_FILENAME, SAMPLE_RATE, myrecording)  # Save as WAV file
    
    from deepspeech_generator import speech_to_text
    transcription = speech_to_text(join(args.deepspeech, "output_graph.pbmm"), join(args.deepspeech, "alphabet.txt"), join(args.deepspeech, "lm.binary"), join(args.deepspeech, "trie"), TMP_FILENAME)
    print(transcription)
    
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
    
    linguistic_model = AttentionModel(linguistic_cfg)
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
    ensemble_model = EnsembleModel(ensemble_cfg)
    ensemble_model.float().to("cpu")
    
    try:
        ensemble_model.load_state_dict(torch.load(args.ensemble_model))
    except:
        print("Failed to load model from {} without device mapping. Trying to load with mapping to {}".format(
            args.ensemble_model, "cpu"))
        ensemble_model.load_state_dict(torch.load(args.ensemble_model, map_location="cpu"))
    
    ensemble_model.eval()
    
    spectrogram = calculate_spectrogram(TMP_FILENAME, False)
    spectrogram = np.expand_dims(spectrogram, axis=0)
    
    preprocessed_transcription = Preprocessor.preprocess_one(transcription)
    transcriptions_emb = Word2VecWrapper.get_sentence_embedding(preprocessed_transcription, 30)
    transcriptions_emb = np.expand_dims(transcriptions_emb, axis=0)
    
    with torch.no_grad():
        output = acoustic_model(spectrogram)
        probabilities = F.softmax(output).numpy()
        print("Acoustic output: {}, probabilities: {}".format(ID_TO_CLASS[np.argmax(output.numpy())], probabilities))
    
        output = linguistic_model(transcriptions_emb)
        ex = linguistic_model.extract(transcriptions_emb)
        probabilities = F.softmax(output).numpy()
        print("Linguistic output: {}, probabilities: {}".format(ID_TO_CLASS[np.argmax(output.numpy())], probabilities))
    
        output = ensemble_model(spectrogram, transcriptions_emb)
        probabilities = F.softmax(output).numpy()
        print("Ensemble output: {}, probabilities: {}".format(ID_TO_CLASS[np.argmax(output.numpy())], probabilities))


