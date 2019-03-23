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


# Linguistic model performance on validation set

Accuracy: **61.87%**  
Weighted Accuracy: **62.75%**  
Confusion matrix:
```
 [[318. 119.  74.  45.]
  [ 90. 265.  30.  21.]
  [ 66.  33. 164.  14.]
  [ 44.  27.   9. 181.]]
```
*classes in order: [Neutral, Happiness, Sadness, Anger]  
*row - correct class, column - prediction
