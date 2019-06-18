import os
import random
import pywt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

from scipy import io

import statistics

from nolds import *
from sklearn.metrics import auc

import heartpy as hp
from heartpy.exceptions import BadSignalWarning

import nitime.algorithms.autoregressive as AR


class BadFeatureGenerationException(Exception):
    pass


def get_heartbeats(ecg, hz=360, constant=90):
    import heartpy
    try:
        if type(ecg) is list:
            w, m = heartpy.process(pd.Series(ecg), hz)
        else:
            w, m = heartpy.process(ecg, hz)
    except BadSignalWarning:
        raise BadFeatureGenerationException()

    peaklist = w["peaklist_cor"]
    heartbeats = []
    for i in range(len(peaklist) - 1):
        if i == 0 and peaklist[i] - constant < 0:
            continue
        ii = peaklist[i] - constant
        jj = ii + 300
        heartbeats.append(ecg[ii:jj])
    return heartbeats, w, m


def ApEn(U, m=2, r=3):
    """
    :returns approximate entropy (wikipedia)
    """

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)
    return abs(_phi(m + 1) - _phi(m))


def calculate_stats_signal(signal, name=""):
    """Returns dict with stats of signal"""
    op = dict()
    name = name + "_" if len(name) > 0 else name
    op[name + "auc"] = auc(list(range(0, len(signal))), signal)
    op[name + "mean"] = np.mean(signal)
    op[name + "var"] = np.var(signal)
    op[name + "stddev"] = np.std(signal)
    return op


class FeatureExtraction:

    @staticmethod
    def generate_features(ecg_signal, hz=360):
        """

        :param ecg_signal: raw signal
        :param hz: Frequency of the signal
        :return: features (list), feature_names (list)
        """
        features_list = list()
        time = len(ecg_signal) / hz
        heartbeats, working_data, measures = get_heartbeats(ecg_signal, hz=hz)
        FeatureExtraction._AR_features(heartbeats=heartbeats, hz=hz, features_list=features_list)
        for (name, stat) in calculate_stats_signal(ecg_signal).items():
            features_list.append((stat, name))
        FeatureExtraction._wavelet_stats(heartbeats=heartbeats, hz=hz, features_list=features_list)  # add all stat
        FeatureExtraction._pan_tompkins(working_data=working_data, measures=measures, hz=hz,
                                        feature_list=features_list)  # add all general data
        try:
            features_list.append((sampen(heartbeats[0]), "sampen"))
            features_list.append((lyap_e(ecg_signal), "lyap_e"))
            features_list.append((lyap_r(ecg_signal), "lyap_r"))
        except Exception:
            raise BadFeatureGenerationException()

        f_fnames = [list(t) for t in zip(*features_list)]
        return f_fnames[0], f_fnames[1]

    @staticmethod
    def _pan_tompkins(ecg=None, hz=360, feature_list=None, working_data=None, measures=None):
        if feature_list is None:
            feature_list = list()
        try:
            # working_data, measures = hp.process(ecg, hz) \
            #                              if working_data is None and measures is None else working_data, measures
            bpm = measures["bpm"]
            if bpm > 300 or np.isnan(bpm):
                # hp.plotter(working_data, measures)
                raise BadFeatureGenerationException()
            feature_list.append((bpm, "bpm"))

            for k in ["ibi", "sdnn", "sdsd", "pnn20", "pnn50", "rmssd", "hr_mad"]:
                if measures[k] is not None and not np.isnan(measures[k]):
                    feature_list.append((np.float32(measures[k]), k))
                else:
                    # print(measures[k])
                    raise BadFeatureGenerationException()
        except Exception:
            raise BadFeatureGenerationException()

    @staticmethod
    def _wavelet_stats(ecg_signal=None, hz=360, features_list=None, heartbeats=None):
        if features_list is None:
            features_list = list()
        try:
            # heartbeats = get_heartbeats(ecg_signal, hz) if heartbeats is None else heartbeats
            dwt_ = pywt.wavedec(heartbeats[0], "db4", level=4, mode="periodic")[0]
            # for i in range(len(dwt_)):
            #     features_list.append((dwt_[i], "dwt_" + str(i)))
            for (name, stat) in calculate_stats_signal(dwt_, name="wavelet").items():
                features_list.append((stat, name))
        except Exception:
            raise BadFeatureGenerationException()

    @staticmethod
    def _AR_features(ecg_signal=None, hz=360, features_list=None, heartbeats=None):
        if features_list is None:
            features_list = list()
        try:
            # heartbeats = get_heartbeats(ecg_signal, hz) if heartbeats is None else heartbeats
            coeff = AR.AR_est_YW(heartbeats[0], 4)[0]
            for i in range(len(coeff)):
                features_list.append((coeff[i], "ar_" + str(i)))
            # features_list.append((sampen(hbs[0]), "sampen_hb"))

        except Exception:
            raise BadFeatureGenerationException()
