import datetime
import json
import random
import string
from os.path import join
import sys

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import *
from sklearn.utils.multiclass import unique_labels




class DictQuery(dict):
    def get(self, path, default=None):
        keys = path.split(".")
        val = None

        for key in keys:
            if val:
                if isinstance(val, list):
                    val = [v.get(key, default) if v else None for v in val]
                else:
                    val = val.get(key, default)
            else:
                val = dict.get(self, key, default)

            if not val:
                break

        return val

    def nested_set(self, path, value=None):
        keys = path.split(".")
        for key in keys[:-1]:
            if len(key) > 0:
                self = self.setdefault(key, {})
        self[keys[-1]] = value


class OutputClassification:
    """
    A class for writing metrics and data of a classifier model to a JSON.
    """
    def __init__(self, name, class_type, output_dir=None, file_dir=None, random_state="", test_size=""):
        self.output = DictQuery({
            "info": {
                "name": name,
                "datetime": str(datetime.datetime.now()),
                "random_state": random_state,
                "test_size": test_size
            },
            "parameters": {
                "features": {
                },
                "model": {
                    "type": class_type,
                }},
            "performance": {

            },
            "data": {
                "positive": {
                    "data": [],
                    "length": 0
                },
                "negative": {
                    "data": [],
                    "length": 0
                }
            },
            "files": []
        })
        self._output_dir = output_dir
        self._file_dir = file_dir

    @property
    def file_dir(self):
        return join(self._output_dir, self._file_dir)

    @property
    def output_dir(self):
        return self._output_dir

    def _plot_confusion_matrix(self, y_true, y_pred, classes,
                               normalize=False,
                               title=None,
                               cmap=plt.cm.Blues):
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
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = [0, 1]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # print("Normalized confusion matrix")
        else:
            pass
            # print('Confusion matrix, without normalization')

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

        def _random_string(length=10):
            """Generate a random string of fixed length """
            letters = string.ascii_lowercase
            return ''.join(random.choice(letters) for i in range(length))

        name = self._file_dir + _random_string(7)
        plt.savefig(name)
        self.add_file(name + ".png", "Confusion Matrix Normalised: %s" % str(normalize))
        return ax

    def generate_json(self, filename=None):
        if filename:
            file = open(join(self._output_dir, filename), "w+")
            json.dump(self.output, file, indent=True)
            file.close()
        else:
            return json.dumps(self.output)

    def add_info(self, path, value):
        self.output.nested_set("info." + path, value)

    def add_model_parameter(self, path, value):
        self.output.nested_set("parameters.model." + path, value)

    def add_feature_parameter(self, path, value):
        self.output.nested_set("parameters.features." + path, value)

    def add_performance(self, path, value):
        self.output.nested_set("performance." + path, value)

    def add_file(self, file_name, description, only=True):
        if "images" not in self.output:
            self.output["images"] = list()
        self.output["images"].append({
            "name": file_name,
            "description": description,
            "only": only
        })

    def fill_multiple_metrics(self, y_real, y_pred):
        self.add_performance("accuracy", accuracy_score(y_real, y_pred))

    def fill_binary_metrics(self, y_real, y_pred, path="", plot=False, y_prob=None):
        precision, recall, fbeta, support = precision_recall_fscore_support(y_real, y_pred, average="binary")
        self.add_performance(path + ".precision", precision)
        self.add_performance(path + ".recall", recall)
        self.add_performance(path + ".f_beta", fbeta)
        self.add_performance(path + ".accuracy", accuracy_score(y_real, y_pred))
        if y_prob is not None:
            self.add_performance(path + ".roc_auc", roc_auc_score(y_real, y_prob[:, 1]))
        self.add_performance(path + ".average_precision", average_precision_score(y_real, y_pred))
        self.add_performance(path + ".jaccard", jaccard_similarity_score(y_real, y_pred))
        self.add_performance(path + ".confusion_matrix",
                             [list(int(j) for j in i) for i in confusion_matrix(y_real, y_pred)])
        # if plot:
        #     self._plot_confusion_matrix(y_real, y_pred, ["Healty", "Sick"], normalize=True)
        #     self._plot_confusion_matrix(y_real, y_pred, ["Healty", "Sick"], normalize=False)

    def add_data(self, name, length=None, positive=False):
        if length is None:
            if positive:
                self.output["data"]["positive"]["data"].append(name)
                self.output["data"]["positive"]["length"] += 1
            else:
                self.output["data"]["negative"]["data"].append(name)
                self.output["data"]["negative"]["length"] += 1
        else:
            if positive:
                self.output["data"]["positive"]["data"] = name
                self.output["data"]["positive"]["length"] = length
            else:
                self.output["data"]["negative"]["data"] = name
                self.output["data"]["negative"]["length"] = length
