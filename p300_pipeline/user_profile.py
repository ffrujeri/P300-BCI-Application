from enum import Enum
import numpy
import os
import pickle
from termcolor import cprint
from user_profile_util import RawDataReader, SpatialFilters, NaiveBayesModel
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import warnings


class ClassifierType(Enum):
    svm = 1
    gaussianNB = 2
    bernoulliNB = 4
    lda = 5
    qda = 6
    knn = 7
    rf = 8
    mlp = 9


class UserProfile:
    USER_PROFILE_DIR = '../user_profile/'
    PARAMS_FILENAME = 'user_profile.params'
    DEFAULT_USER_ID = 'player1'

    def __init__(self):
        self.id = self.DEFAULT_USER_ID
        self.optimal_num_filters = 0
        self.spatial_filters = None
        self.classifier = None
        self.classifier_type = None
        self.epoch_duration = 0
        self.epoch_size = 0
        self.num_epochs = 0

        self.accuracy = -1
        self.precision = -1
        self.recall = -1
        self.f1_score = -1
        self.target_mean = self.non_target_mean = 0
        self.target_variance = self.non_target_variance = 0

    def fetch_profile(self, profile_filename):
        try:
            learning_params_path = self.get_learning_params_path(
                profile_filename)
            file_to_read = open(learning_params_path, 'r')
            self_dict = pickle.load(file_to_read)
            file_to_read.close()
            self.__dict__.update(self_dict)
        except:  # TODO: specify exception; comment
            self.compute_profile(profile_filename)

    def save_profile_params_file(self, profile_filename):
        params_path = UserProfile.get_learning_params_path(profile_filename)
        output_file = open(params_path, 'w')
        pickle.dump(self.__dict__, output_file)
        output_file.close()

    @staticmethod
    def get_user_profile_path(profile_filename):
        return os.path.join(UserProfile.USER_PROFILE_DIR, profile_filename)

    @staticmethod
    def get_learning_params_path(profile_filename):
        return os.path.join(UserProfile.get_user_profile_path(profile_filename),
                            UserProfile.PARAMS_FILENAME)

    def compute_profile(self, profile_filename,
                        classifier_types=(ClassifierType.bernoulliNB, ClassifierType.gaussianNB, ClassifierType.lda),
                        epoch_duration=0.7,
                        num_filters=None,
                        test_size=None,
                        equal_labels_ratio=True):  # TODO
        self.epoch_duration = epoch_duration

        # read raw data (raw signal and triggers)
        user_profile_path = self.get_user_profile_path(profile_filename)
        reader = RawDataReader(user_profile_path)
        eeg_signal, triggers = reader.read_raw_data()

        # process signal (bandpass filter, spatial filters, epoching)
        eeg_signal.set_triggers(triggers)
        eeg_signal.apply_bandpass_filter()
        eeg_signal.apply_spatial_filters(self.epoch_duration)
        eeg_signal.epoch_virtual_signal()
        self.spatial_filters = eeg_signal.spatial_filters
        self.epoch_size = eeg_signal.epoch_size
        self.num_epochs = len(triggers.indexes)
        self.compute_means_and_variances(eeg_signal, triggers)

        # select optimal number of filters and model
        current_best = 0
        if num_filters:
            min_filters = num_filters
            max_filters = num_filters + 1
        else:
            min_filters = 1
            max_filters = SpatialFilters.MAX_NUM_FILTERS + 1
        for classifier_type in classifier_types:
            for num_filters in range(min_filters, max_filters):
                concatenated_epochs = SpatialFilters.concatenate_channels(eeg_signal.epoched_signals[:num_filters])
                X = concatenated_epochs
                y = eeg_signal.triggers.labels
                if equal_labels_ratio:
                    X, y = self.truncate(X, y)
                clf = self.get_classifier(classifier_type)
                f1_score = numpy.average(cross_val_score(clf, X, y, cv=5, scoring='f1'))
                accuracy = numpy.average(cross_val_score(clf, X, y, cv=5, scoring='accuracy'))
                precision = numpy.average(cross_val_score(clf, X, y, cv=5, scoring='average_precision'))
                recall = numpy.average(cross_val_score(clf, X, y, cv=5, scoring='recall'))

                current = f1_score
                if current > current_best:
                    current_best = current
                    self.f1_score = f1_score
                    self.accuracy = accuracy
                    self.precision = precision
                    self.recall = recall
                    self.optimal_num_filters = num_filters
                    self.classifier_type = classifier_type

        concatenated_epochs = SpatialFilters.concatenate_channels(eeg_signal.epoched_signals[:self.optimal_num_filters])
        X = concatenated_epochs
        y = eeg_signal.triggers.labels
        self.classifier = self.compute_classifier(self.classifier_type, X, y)

        self.save_profile_params_file(profile_filename)

    @staticmethod
    def truncate(features, labels):
        """
        Returns new features and labels with equal distribution of labels
        and maximum size.
        """
        target = numpy.array(features)[numpy.array(labels) == True]
        non_target = numpy.array(features)[numpy.array(labels) == False]
        max_len = numpy.min([len(target), len(non_target)])

        new_features = []
        new_labels = []
        target = 0
        non_target = 0
        for i in range(len(features)):
            if labels[i] and target < max_len:
                target += 1
                new_features.append(features[i])
                new_labels.append(labels[i])
            elif not labels[i] and non_target < max_len:
                non_target += 1
                new_features.append(features[i])
                new_labels.append(labels[i])
        return new_features, new_labels

    @staticmethod
    def compute_classifier(classifier_type, X_train, y_train):
        if classifier_type == ClassifierType.svm:
            return svm.SVC(kernel='linear').fit(X_train, y_train)
        elif classifier_type == ClassifierType.gaussianNB:
            return GaussianNB().fit(X_train, y_train)
        elif classifier_type == ClassifierType.bernoulliNB:
            return BernoulliNB().fit(X_train, y_train)
        elif classifier_type == ClassifierType.lda:
            return LinearDiscriminantAnalysis().fit(X_train, y_train)
        elif classifier_type == ClassifierType.qda:
            return QuadraticDiscriminantAnalysis().fit(X_train, y_train)
        elif classifier_type == ClassifierType.knn:
            return KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
        elif classifier_type == ClassifierType.rf:
            return RandomForestClassifier(n_estimators=2).fit(X_train, y_train)
        elif classifier_type == ClassifierType.mlp:
            return MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5), random_state=1).fit(X_train, y_train)
        else:
            warnings.warn('Invalid classifier type. Using default.')
            return svm.SVC(kernel='linear').fit(X_train, y_train)

    @staticmethod
    def get_classifier(classifier_type):
        if classifier_type == ClassifierType.svm:
            return svm.SVC(kernel='linear')
        elif classifier_type == ClassifierType.gaussianNB:
            return GaussianNB()
        elif classifier_type == ClassifierType.bernoulliNB:
            return BernoulliNB()
        elif classifier_type == ClassifierType.lda:
            return LinearDiscriminantAnalysis()
        elif classifier_type == ClassifierType.qda:
            return QuadraticDiscriminantAnalysis()
        elif classifier_type == ClassifierType.knn:
            return KNeighborsClassifier(n_neighbors=3)
        elif classifier_type == ClassifierType.rf:
            return RandomForestClassifier(n_estimators=2)
        elif classifier_type == ClassifierType.mlp:
            return MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1)
        else:
            warnings.warn('Invalid classifier type. Using default.')
            return BernoulliNB()

    def compute_means_and_variances(self, eeg_signal, triggers):
        training_features = SpatialFilters.concatenate_channels(
            eeg_signal.epoched_signals[:1])  # SpatialFilters.MAX_NUM_FILTERS
        labels = triggers.labels

        target_features = training_features[labels == True]  # TODO: fix warning
        non_target_features = training_features[labels == False]

        num_target_features, num_target_samples = target_features.shape
        num_no_target_features, num_no_target_samples = non_target_features.shape

        self.non_target_mean = non_target_features.mean(0)
        V1 = (non_target_features - self.non_target_mean) ** 2
        self.non_target_variance = V1.sum(0) / float(num_no_target_features - 1)

        self.target_mean = target_features.mean(0)
        V2 = (target_features - self.target_mean) ** 2
        self.target_variance = V2.sum(0) / float(num_target_features - 1)

    def print_model_info(self):
        cprint('model params', 'yellow', attrs=['bold'])
        cprint('> classifier used: {}'.format(self.classifier_type), 'yellow')
        cprint('> accuracy = {:.2f}'.format(self.accuracy * 100), 'yellow')
        cprint('> num_filters = {}'.format(self.optimal_num_filters), 'yellow')
        cprint('> precision = {}'.format(self.precision), 'yellow')
        cprint('> P(p300 | label = p300) = {}'.format(self.precision), 'yellow')
        aux = (1/self.recall - 1) / (1 / (1 - self.accuracy) * (1/self.precision + 1/self.recall + self.accuracy - 3))
        nvp = 1 / (1 + aux)
        cprint('> P(not p300 | label = not p300) = {}'.format(nvp), 'yellow')

    def compute_likelihoods(self, epoch):  # TODO: check
        spatial_filters = self.spatial_filters.transpose()[:self.optimal_num_filters, :]
        virtual_epoch = numpy.dot(spatial_filters, epoch)
        cprint('USER_MODEL process_new_epoch xdawned_new_erp = ' + str(numpy.size(virtual_epoch)), 'magenta', attrs=['bold'])
        virtual_epoch = virtual_epoch.reshape(-1)
        return self.classifier.predict(virtual_epoch)
