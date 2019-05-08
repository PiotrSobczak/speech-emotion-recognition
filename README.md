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

# System Architecture
<img src="https://github.com/PiotrSobczak/speech-emotion-recognition/blob/master/assets/ensemble_arch_live.png" width="600"></img>

# Results so far

| Model | Accuracy | Unweighted Accuracy  | Loss  |
|---|---|---|---|
| Acoustic  |  0.602 | 0.601  | 0.983  |
| Linguistic  | 0.642  | 0.638  | 0.913  |
| Ensemble (highest confidence) | 0.699  | 0.704  | 0.827  |
| Ensemble (average) | 0.711  | 0.708  | 0.948  |
| Ensemble (weighted average) | 0.716  | 0.712  | 0.944  |

### Confusion matrix of the best model
```
loss: 0.944, acc: 0.716. unweighted acc: 0.712, conf_mat: 
[[291.  60.  31.   9.]
 [ 88. 282.  17.   6.]
 [ 46.  19. 191.   2.]
 [ 61.  26.   4. 167.]]
```

*classes in order: [Neutral, Happiness, Sadness, Anger]  
*row - correct class, column - prediction
