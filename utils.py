import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import interp
from sklearn.metrics import roc_curve, auc, confusion_matrix


def predict_from_proba(clf, proba):
    """
    From SciPy
    :param clf:
    :param proba:
    :return: predictions from the proba.
    """
    if clf.n_outputs_ == 1:
        return clf.classes_.take(np.argmax(proba, axis=1), axis=0)

    else:
        n_samples = proba[0].shape[0]
        predictions = np.zeros((n_samples, clf.n_outputs_))

        for k in range(clf.n_outputs_):
            predictions[:, k] = clf.classes_[k].take(np.argmax(proba[k],
                                                               axis=1),
                                                     axis=0)
        return predictions


def matrix_in_list(matrix):
    output = list()
    for l in matrix:
        for ll in l:
            output.append(ll)
    return output


def make_cool_roc_curve(y_tests, y_probas, plot_folds=True, title=""):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for y_test, probas_ in zip(y_tests, y_probas):
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        if plot_folds:
            plt.plot(fpr, tpr, lw=1, alpha=0.5,
                     label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey',
             label='Chance', alpha=.5)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    label = r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc) if plot_folds else \
        r'ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc)
    plt.plot(mean_fpr, mean_tpr,
             label=label,
             lw=2, alpha=.9)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    if plot_folds:
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver operating characteristic {title}')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(y_true=None, y_pred=None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, cm=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix

    if cm is not None:
        cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = [0, 1]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
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
    plt.show()
    return ax

def make_cool_roc_curve2(y_tests, y_probas, plot_folds=False, filename="ROC"):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for y_test, probas_ in zip(y_tests, y_probas):
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        if plot_folds:
            plt.plot(fpr, tpr, lw=1, alpha=0.5,
                     label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey',
             label='Chance', alpha=.5)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    label = r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc) if plot_folds else \
        r'ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc)
    plt.plot(mean_fpr, mean_tpr, color="black",
             label=label,
             lw=2, alpha=.9)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    if plot_folds:
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filename + ".png")