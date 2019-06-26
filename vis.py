import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_linguistic_dataset, CLASS_TO_ID

test_features, test_labels, val_features, val_labels, train_features, train_labels = load_linguistic_dataset()
print(
    "Subsets sizes: test_features:{}, test_labels:{}, val_features:{}, val_labels:{}, train_features:{}, train_labels:{}".format(
        test_features.shape[0], test_labels.shape[0], val_features.shape[0], val_labels.shape[0],
        train_features.shape[0], train_labels.shape[0])
)

plt.bar(np.arange(3), [train_labels.shape[0], test_labels.shape[0], val_labels.shape[0]], align='center', alpha=0.5)
plt.xticks(np.arange(3), ["training", "test", "validation"])
plt.ylabel('Number of samples')
plt.title("Dataset split")
plt.show()

class_distr = [0, 0, 0, 0]

for name, labels in zip(["Test", "Validation", "Training"], [test_labels, val_labels, train_labels]):
    print(name)
    for i in range(4):
        print("{}: {}".format(i, labels[labels == i].shape[0]))
        class_distr[i] += labels[labels == i].shape[0]
    subset_sizes = [labels[labels == i].shape[0] for i in range(4)]

    plt.bar(np.arange(len(subset_sizes)), subset_sizes, align='center', alpha=0.5)
    plt.xticks(np.arange(len(subset_sizes)), CLASS_TO_ID.keys())
    plt.ylabel('Number of samples')
    plt.title(name + " set")

    plt.show()

plt.bar(np.arange(4), class_distr, align='center', alpha=0.5)
plt.xticks(np.arange(4), CLASS_TO_ID.keys())
plt.ylabel('Number of samples')
plt.title("Class distribution")
plt.show()