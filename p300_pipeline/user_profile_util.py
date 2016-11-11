from neo.io import RawBinarySignalIO
import numpy as np
import os
import pickle
import quantities
from scipy.linalg import toeplitz, qr, inv, svd
import scipy.signal
from sklearn import svm
from termcolor import cprint
import yaml


class RawDataReader:
    JSON_SUFFIX = 'info.json'
    STREAMS_ID = 'streams'
    STREAM_TYPE_ID = 'stream_type'
    TRIGGERS_STREAM_ID = 'AsynchronusEventStream'  # typo intended
    ANALOG_SIGNAL_STREAM_ID = 'AnalogSignalSharedMemStream'

    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path

    def read_raw_data(self):
        """Reads data from info.json."""
        info_path = os.path.join(self.raw_data_path, self.JSON_SUFFIX)
        info = yaml.safe_load(open(info_path))  # read strings as string objects

        eeg_signal = None
        triggers = None
        for stream_info in info[self.STREAMS_ID]:
            if stream_info[self.STREAM_TYPE_ID] == self.ANALOG_SIGNAL_STREAM_ID:
                eeg_signal = self._read_raw_signal(stream_info)
            elif stream_info[self.STREAM_TYPE_ID] == self.TRIGGERS_STREAM_ID:
                triggers = self._read_triggers(stream_info)

        if eeg_signal is None:
            raise RuntimeError(self.ANALOG_SIGNAL_STREAM_ID +
                               ' not found in info.json')
        if triggers is None:
            raise RuntimeError(self.TRIGGERS_STREAM_ID +
                               ' not found in info.json')

        return eeg_signal, triggers

    def _read_raw_signal(self, stream_info):
        if stream_info['name'] == '':
            raise RuntimeError('Raw signal data file not found.')
        path = os.path.join(self.raw_data_path, stream_info['name']) + '.raw'
        reader = RawBinarySignalIO(path)
        segment = reader.read_segment(
            sampling_rate=stream_info['sampling_rate'] * quantities.kHz,
            t_start=0.*quantities.s,
            unit=quantities.V,
            nbchannel=stream_info['nb_channel'],
            dtype=stream_info['dtype']
        )
        sampling_frequency = (segment.analogsignals[0].sampling_rate.item())
        signal = np.array(segment.analogsignals)
        cprint('Read raw signal from ' + path, 'magenta')
        return EEGSignal(signal, sampling_frequency)

    def _read_triggers(self, stream_info):
        if stream_info['name'] == '':
            raise RuntimeError('Triggers raw data file not found.')
        path = os.path.join(self.raw_data_path, stream_info['name']) + '.raw'
        triggers_dtype = [tuple(item) for item in stream_info['dtype']]
        triggers = np.memmap(filename=path, mode='r', dtype=triggers_dtype)
        labels = np.array(triggers['code']) == 2  # 1: non target; 2: target
        indexes = np.rint(np.array(triggers['pos'])) - 1
        cprint('Read {} triggers from {}'.format(len(indexes), path), 'magenta')
        return Triggers(labels, indexes)


class EEGSignal:
    EPOCH_DURATION = .875

    def __init__(self, raw_signals, sampling_frequency):
        self.raw_signals = raw_signals
        self.sampling_frequency = sampling_frequency
        self.num_channels = raw_signals.shape[0]
        self.epoch_size = int(self.EPOCH_DURATION * self.sampling_frequency)

        self.filtered_signals = None
        self.triggers = None
        self.epoched_signals = None
        self.spatial_filters = None
        self.virtual_signals = None

    def set_triggers(self, triggers):
        self.triggers = triggers

    def apply_bandpass_filter(self, filter_order=2, low_freq=.5, high_freq=20):
        self.filtered_signals = np.empty_like(self.raw_signals)

        wn = (low_freq, high_freq) / np.array(self.sampling_frequency / 2)
        butter_filter = scipy.signal.butter(N=filter_order, Wn=wn, btype='band')
        for index, signal in enumerate(self.raw_signals):  # for each channel
            self.filtered_signals[index] = scipy.signal.lfilter(
                butter_filter[0], butter_filter[1], signal)

    def apply_spatial_filters(self):
        sp = SpatialFilters(self.filtered_signals, self.triggers)
        sp.compute_spatial_filters(self.epoch_size)
        self.spatial_filters = sp.spatial_filters
        self.virtual_signals = self.spatial_filters.T.dot(self.filtered_signals)

    def epoch_virtual_signal(self, triggers=None, remove_artifacts_flag=True):
        if triggers:
            self.triggers = triggers
        epoched_signals = list()
        for signal_channel in self.virtual_signals:
            epochs = list()
            for index in self.triggers.indexes:
                epochs.append(signal_channel[index:(index + self.epoch_size)])
            epoched_signals.append(epochs)
        self.epoched_signals = np.array(epoched_signals)
        if remove_artifacts_flag:  # remove artifacts and re-epoch
            self._remove_artifacts()

    def _remove_artifacts(self, reject_above=0.9):
        epochs_max_value = self.epoched_signals.max(2).max(0)
        threshold_index = int(epochs_max_value.size * reject_above)
        threshold_value = np.sort(epochs_max_value)[threshold_index]
        indexes = np.where(epochs_max_value > threshold_value)[0]
        self.triggers.indexes = np.delete(self.triggers.indexes, indexes)
        self.triggers.labels = np.delete(self.triggers.labels, indexes)
        self.epoch_virtual_signal(remove_artifacts_flag=False)  # re-epoch


class Triggers:
    def __init__(self, labels, indexes):
        self.labels = labels
        self.indexes = indexes


class SpatialFilters:
    MAX_NUM_FILTERS = 4

    def __init__(self, filtered_signals, triggers):
        self.filtered_signals = filtered_signals
        self.triggers = triggers
        self.spatial_filters = None

    def compute_spatial_filters(self, epoch_size):
        num_channels, num_samples = self.filtered_signals.shape

        event_indexes = self.triggers.indexes.astype('int')
        d1_col = np.zeros(num_samples)
        d1_col[event_indexes] = self.triggers.labels + 1  # 2: tgt; 1: non tgt
        d1_col[np.nonzero(d1_col)] = 1
        d1_row = np.ones(epoch_size) if d1_col[0] == 1 else np.zeros(epoch_size)
        D1 = toeplitz(d1_col, d1_row)

        d2_col = np.zeros(num_samples)
        d2_col[event_indexes[np.nonzero(self.triggers.labels)]] = 1
        d2_row = np.ones(epoch_size) if d2_col[0] == 1 else np.zeros(epoch_size)
        D2 = toeplitz(d2_col, d2_row)

        D = np.hstack((D1, D2, np.ones((D1.shape[0], 1))))
        DtD = np.dot(D.T, D)
        DtD_inv = inv(DtD)
        DtY = np.dot(D.T, self.filtered_signals.T)
        A = np.dot(DtD_inv, DtY)
        D = D2
        Qy, Ry = qr(self.filtered_signals.astype('float64').T, mode='economic')
        D = D.astype('float64')
        Qd, Rd = qr(D, mode='economic')
        Q = np.dot(Qd.T, Qy)
        U, S, V = svd(Q, full_matrices=True)

        # compute spatial filters
        W = np.zeros((num_channels, self.MAX_NUM_FILTERS))
        for i in range(self.MAX_NUM_FILTERS):
            W[:, i] = np.dot(inv(Ry), V.T[:, i])
        self.spatial_filters = W

    @staticmethod
    def concatenate_channels(epoched_signal):
        num_virtual_channels, num_averaged_epochs, num_samples = \
            epoched_signal.shape
        stab = np.swapaxes(epoched_signal, 0, 1)
        concatenated_epochs = stab.reshape(
            (num_averaged_epochs, num_samples * num_virtual_channels))
        return concatenated_epochs


class Model:
    def __init__(self, training_features=None, labels=None):
        if training_features is None and labels is None:
            training_features = []
            labels = []

        self.training_features = training_features
        self.labels = labels

        if len(training_features) != len(labels):
            raise AssertionError(
                'Training features and labels should have the same size!')

        self.true_positives = self.false_positives = 0
        self.true_negatives = self.false_negatives = 0

        self.accuracy = None
        self.spatial_filters = None
        self.optimal_num_filters = None

    def set_params(self, params_dict):
        raise NotImplementedError

    def set_w_xdawn_params(self, spatial_filters, optimal_num_filters):
        self.spatial_filters = spatial_filters
        self.optimal_num_filters = optimal_num_filters

    def compute_params(self):
        raise NotImplementedError

    def compute_accuracy(self):
        raise NotImplementedError

    def compute_likelihoods(self, feature_vector):
        raise NotImplementedError

    def is_target(self, data):
        raise NotImplementedError

    def save_model_params_file(self, params_path):
        raise NotImplementedError


class NaiveBayesModel:
    def __init__(self, training_features=None, labels=None, prior=0.2):
        if training_features is None and labels is None:
            training_features = []
            labels = []
        self.training_features = training_features
        self.labels = labels
        self.prior = prior

        if len(training_features) != len(labels):
            raise AssertionError(
                'Training features and labels should have the same size!')

        self.m1 = self.v1 = self.t1 = None
        self.m2 = self.v2 = self.t2 = None

        self.true_positives = self.false_positives = 0
        self.true_negatives = self.false_negatives = 0

        self.accuracy = None
        self.spatial_filters = None
        self.optimal_num_filters = None

    def set_prior(self, prior):
        self.prior = prior

    def set_params(self, params_dict):
        self.prior = params_dict['prior']

        self.m1 = params_dict['m1']
        self.v1 = params_dict['v1']
        self.t1 = params_dict['t1']

        self.m2 = params_dict['m2']
        self.v2 = params_dict['v2']
        self.t2 = params_dict['t2']

        self.accuracy = params_dict['accuracy']
        self.spatial_filters = params_dict['w_xdawn']
        self.optimal_num_filters = params_dict['optimal_num_filters']

    def set_w_xdawn_params(self, spatial_filters, optimal_num_filters):
        self.spatial_filters = spatial_filters
        self.optimal_num_filters = optimal_num_filters

    def compute_params(self):
        """Naive Bayes classifier"""
        target_features = self.training_features[self.labels == True]
        no_target_features = self.training_features[self.labels == False]

        num_target_features, num_target_samples = target_features.shape
        num_no_target_features, num_no_target_samples = no_target_features.shape

        self.m1 = no_target_features.mean(0)
        V1 = (no_target_features - self.m1) ** 2
        self.v1 = V1.sum(0) / float(num_no_target_features - 1)
        self.t1 = np.sum(np.log(np.sqrt(self.v1)))

        self.m2 = target_features.mean(0)
        V2 = (target_features - self.m2) ** 2
        self.v2 = V2.sum(0) / float(num_target_features - 1)
        self.t2 = np.sum(np.log(np.sqrt(self.v2)))

    def compute_accuracy(self):
        """Computes accuracy through leave one out cross-validation."""
        total_labels = len(self.labels)

        for index, test_feature in enumerate(self.training_features):
            training_data = np.delete(self.training_features, index, axis=0)
            labels = np.delete(self.labels, index)
            model = NaiveBayesModel(training_data, labels, self.prior)
            model.compute_params()
            if model.is_target(test_feature):  # (TARGET)
                if model.is_target(test_feature) == self.labels[index]:
                    self.true_positives += 1
                else:
                    self.false_positives += 1
            else:  # model.is_target(test_feature) False (NOT TARGET)
                if model.is_target(test_feature) == self.labels[index]:
                    self.true_negatives += 1
                else:
                    self.false_negatives += 1
        self.accuracy = 100. * (self.true_positives +
                                self.true_negatives) / total_labels
        return self.accuracy

    def compute_likelihoods(self, feature_vector):
        vec1 = (feature_vector - self.m1) ** 2
        vec1 = vec1.astype('float') / self.v1
        sum_vec1 = vec1.sum(0)
        vec2 = (feature_vector - self.m2) ** 2
        vec2 = vec2.astype('float') / self.v2
        sum_vec2 = vec2.sum(0)
        lf1 = - (0.5 * sum_vec1 + self.t1)
        lf2 = - (0.5 * sum_vec2 + self.t2)
        return lf1, lf2

    def is_target(self, data):
        lf1, lf2 = self.compute_likelihoods(data)
        delta = lf2 - lf1
        prior = self.prior
        frac_prior = prior / float(1. - prior)
        post = frac_prior * np.exp(delta) / (1 + frac_prior * np.exp(delta))

        post_no_target = post + (1 - 2 * post)
        post_target = 1 - post_no_target

        return post_target > .5

    def save_model_params_file(self, params_path):
        params_dict = {
            'prior': self.prior,
            'm1': self.m1,
            'v1': self.v1,
            't1': self.t1,
            'm2': self.m2,
            'v2': self.v2,
            't2': self.t2,
            'optimal_num_filters': self.optimal_num_filters,
            'w_xdawn': self.spatial_filters,
            'accuracy': self.accuracy
        }

        output_file = open(params_path, 'w')
        pickle.dump(params_dict, output_file)
        output_file.close()


class SVMModel:
    def __init__(self, training_features=None, labels=None):
        if training_features is None and labels is None:
            training_features = []
            labels = []

        self.training_features = training_features
        self.labels = labels

        if len(training_features) != len(labels):
            raise AssertionError(
                'Training features and labels should have the same size!')

        self.true_positives = self.false_positives = 0
        self.true_negatives = self.false_negatives = 0

        self.accuracy = None
        self.spatial_filters = None
        self.optimal_num_filters = None

    def set_params(self, params_dict):
        raise NotImplementedError

    def set_w_xdawn_params(self, spatial_filters, optimal_num_filters):
        self.spatial_filters = spatial_filters
        self.optimal_num_filters = optimal_num_filters

    def compute_params(self):
        raise NotImplementedError

    def compute_accuracy(self, test_set_ratio=0.1):
        raise NotImplementedError

    def compute_likelihoods(self, feature_vector):
        raise NotImplementedError

    def is_target(self, data):
        raise NotImplementedError

    def save_model_params_file(self, params_path):
        raise NotImplementedError
