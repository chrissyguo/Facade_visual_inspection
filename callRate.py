import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(labels, preds, classes, normalize = True, title = None, cmap = plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # compute confusion matrix
    cm = confusion_matrix(labels, preds)
    # Only use the labels that appear in the data
    classes = np.array(classes)[unique_labels(labels, preds)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def compare(list1, n):
    output = []
    for i in range(len(list1)):
        if list1[i] == n:
            output += [1]
        else:
            output += [0]

    return output

def intersection(list1, list2, n):
    output = []
    for i in range(len(list1)):
        if list1[i] == n[0] and list2[i] == n[1]:
            output += [1]
        else:
            output += [0]
    return output



def plot_ROC(labels, preds, outputs, class_names, title = 'ROC curves'):
    n = len(class_names)
    binary_labels = label_binarize(labels, classes=list(range(len(class_names))))
    binary_labels = np.asarray(binary_labels)
    outputs = np.asarray(outputs)

    # indices
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    specificity = dict()
    f1 = dict()

    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i in range(n):
        tp += sum(intersection(compare(preds, i), compare(labels, i), [1, 1]))
        tn += sum(intersection(compare(preds, i), compare(labels, i), [0, 0]))
        fn += sum(intersection(compare(preds, i), compare(labels, i), [0, 1]))
        fp += sum(intersection(compare(preds, i), compare(labels, i), [1, 0]))

        fpr[i], tpr[i], _ = roc_curve(binary_labels[:, i], outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        if tp + fp == 0:
            precision[i] = 0
        else:
            precision[i] = tp / (tp + fp)

        if tp + fn == 0:
            recall[i] = 0
        else:
            recall[i] = tp/(tp+fn)

        if tp + fn == 0:
            specificity[i] = 0
        else:
            specificity[i] = tn/(tn+fp)

        if precision[i] + recall[i] ==0:
            f1[i] = 0
        else:
            f1[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n

    # Compute macro-average ROC curve and ROC area
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(binary_labels.ravel(), outputs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot all ROC curves
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    for i, color in zip(range(n), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='{0} (area = {1:0.2f})'
                       ''.format(class_names[i], roc_auc[i]))


    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, precision, recall, f1, specificity
