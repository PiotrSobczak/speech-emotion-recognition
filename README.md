# What's this project about?
The goal if this project is to create a multi-modal Speech Emotion Recogniton system on [IEMOCAP](https://sail.usc.edu/iemocap/) dataset.

# Project outline
- Feb 2019 - IEMOCAP dataset aquisition and parsing
- Mar 2019 - Baseline of linguistic model
- Apr 2019 - Baseline of acoustic model
- May 2019 - Integration and optimiaztion of both models
- Jun 2019 - Integration with open-source ASR(most likely DeepSpeech)


# What's IEMOCAP dataset?
IEMOCAP states for *Interactive  Emotional  Dyadic  Motion and  Capture* dataset. It is the most popular database used for multi-modal speech emotion recognition.  

**Original class distribution:**  
<img src="https://github.com/PiotrSobczak/speech-emotion-recognition/blob/master/assets/iemocap_original.png" width="300"></img>  

IEMOCAP database suffers from major class imbalance. To solve this problem we reduce the number of classes to 4 and merge *Enthusiastic* and *Happiness* into one class.

**Final class distribution**  
<img src="https://github.com/PiotrSobczak/speech-emotion-recognition/blob/master/assets/iemocap_reorganized.png" width="300"></img>

# Related works overview
<img src="https://github.com/PiotrSobczak/speech-emotion-recognition/blob/master/assets/related_works.png" width="600"></img>

References:
[[1](http://aclweb.org/anthology/C18-1201)]
[[2](https://arxiv.org/pdf/1810.04635.pdf)]
[[3](https://arxiv.org/pdf/1804.05788.pdf)]
[[4](https://arxiv.org/abs/1802.08332)]
[[5](https://arxiv.org/abs/1802.05630)]
[[6](http://dpi-proceedings.com/index.php/dtcse/article/view/17273/16777)]
[[7](https://arxiv.org/pdf/1706.00612.pdf)]
[[8](https://kundoc.com/queue/pdf-evaluating-deep-learning-architectures-for-speech-emotion-recognition-.html)]
[[9](http://www.utdallas.edu/~mirsamadi/files/mirsamadi17a.pdf)]

# Tested Architectures

### Acoustic Architectures

| Classifier Architecture | Input type | Accuracy [%] |
|---|---|---|
| Convolutional Neural Network | Spectrogram | 55.3 
| Bidirectional LSTM with self-attention | LLD features  | 53.2  |

### Linguistic Architectures

| Classifier Architecture | Input type | Accuracy[%] |
|---|---|---|
| LSTM | Transcription | 58.9 | 
| Bidirectional LSTM | Transcription  | 59.4 |
| Bidirectional LSTM with self-attention | Transcription  | 63.1  |

### Ensemble Architectures
Ensemble architectures make use of the most accurate acoustic and linguistic architectures. This means that linguistic model with bidirectional LSTM with self-attention architecture and acoustic model with Convolutional architecture are being used.

| Ensemble type |  Accuracy |
|---|---|
| Decision-level Ensemble(maximum confidence) | 66.7 |
| Decision-level Ensemble(average)  | 68.8 |
| Decision-level Ensemble(weighted average)  | 69.0 |
| Feature-level Ensemble  | 71.1 |

# Feature-level Ensemble Architecture  
<img src="https://github.com/PiotrSobczak/speech-emotion-recognition/blob/master/assets/feature_ensemble_arch.png" width="600"></img>

# Feature-level Ensemble Confusion Matrix
<img src="https://github.com/PiotrSobczak/speech-emotion-recognition/blob/master/assets/confusion_matrix.png" width="600"></img>


# How to prepare IEMOCAP dataset?
- 1.Download IEMOCAP dataset from https://sail.usc.edu/iemocap/
- 2.Create dataset pickle using this module:  
https://github.com/didi/delta/blob/master/egs/iemocap/emo/v1/local/python/mocap_data_collect.py 
- 3.Use create_balanced_iemocap() to get balanced version of iemocap dataset containing 4 classes
- 4.Use load_<DATASET_TYPE>_dataset to load a specific dataset.   
*The first time you load datasets, they will be created from scratch and cached in .npy files. This might take a while.*     
*Next time you load datasets, they will be loaded from cached .npy files*   


# How to run? 
### Run hyperparameter tuning
```
python3 -m speech_emotion_recognition.run_hyperparameter_tuning -m acoustic-spectrogram
```
### Run training
```
python3 -m speech_emotion_recognition.run_training_ensemble -m acoustic-spectrogram
```
### Run ensemble training
```
python3 -m speech_emotion_recognition.run_training_ensemble -a /path/to/acoustic_spec_model.torch -l /path/to/linguistic_model.torch
```
### Run evaluation
```
python3 -m speech_emotion_recognition.run_evaluate -a /path/to/acoustic_spec_model.torch -l /path/to/linguistic_model.torch -e /path/to/ensemble_model.torch
```

# How to run in docker?(CPU only)
### Run hyperparameter tuning
```
docker run -t -v /path/to/project/data:/data -v /path/to/project/saved_models:/saved_models -v /tmp:/tmp speech-emotion-recognition -m speech_emotion_recognition.run_hyperparameter_tuning -m acoustic-spectrogram
```
### Run training
```
docker run -t -v /path/to/project/data:/data -v /path/to/project/saved_models:/saved_models -v /tmp:/tmp speech-emotion-recognition -m speech_emotion_recognition.run_training_ensemble -m acoustic-spectrogram
```
### Run ensemble training
```
docker run -t -v /path/to/project/data:/data -v /path/to/project/saved_models:/saved_models -v /tmp:/tmp speech-emotion-recognition -m speech_emotion_recognition.run_training_ensemble -a /path/to/acoustic_spec_model.torch -l /path/to/linguistic_model.torch
```
### Run evaluation
```
docker run -t -v /path/to/project/data:/data -v /path/to/project/saved_models:/saved_models -v /tmp:/tmp speech-emotion-recognition -m speech_emotion_recognition.run_evaluate -a /path/to/acoustic_spec_model.torch -l /path/to/linguistic_model.torch -e /path/to/ensemble_model.torch
```
