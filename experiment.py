from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from random import shuffle

# Abstract class for experiments on data
class Experiment(object):

    def __init__(self, model, criterion, features, labels):

        self.features = features
        self.labels = labels
        self.model = model
        self.criterion = criterion

        self.last_test_idx = None
        self.last_train_idx = None

    # Split up my data into training and testing
    # Can stick shuffles, stratification, etc. in here.
    def partition(self, features, labels): raise NotImplementedError

    # Run the experiment and get back a  a training and testing validation, preferably averaged across folds
    def run(self):

        train_idx, test_idx = self.partition(self.features, self.labels)

        self.last_test_idx = test_idx
        self.last_train_idx = train_idx

        print("%d TRAIN SAMPLES \t %d TEST SAMPLES" % (len(train_idx), len(test_idx)))

        train_features, train_labels = self.features[train_idx, :], self.labels[train_idx]
        test_features, test_labels = self.features[test_idx, :], self.labels[test_idx]

        self.model.fit(train_features, train_labels)

        train_metric = self.criterion(train_labels, self.model.predict(train_features))
        test_metric = self.criterion(test_labels, self.model.predict(test_features))

        return train_metric, test_metric

# Class for performing a stratified shuffle for a single experimental run
class StratifiedExperiment(Experiment):

    def __init__(self, model, criterion, features, labels, test_size=0.2):

        self.test_size = test_size

        super(StratifiedExperiment, self).__init__(model, criterion, features, labels)

    # In this partition I'm doing a stratified shuffle
    def partition(self, features, labels):

        shuf = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size)
        splits = shuf.split(features, labels)

        for train_idx, test_idx in splits:
            return train_idx, test_idx




# Class for doing a balanced class experiment instead
# I only really made it work for binary label problems just for this presentation though :/
class BinaryBalancedExperiment(Experiment):

    def __init__(self, model, criterion, features, labels, test_size=0.2):

        self.test_size = test_size

        super(BinaryBalancedExperiment, self).__init__(model, criterion, features, labels)


    # In this partition I balance classes and even go as far as to throw out additional data if it causes imbalance.
    # This is to avoid exposing the model to a strong prior distribution in the data
    def partition(self, features, labels):

        # First find instances of each label, as just a pure idx number
        positives = [idx for idx, label in enumerate(labels) if label > 0]
        negatives = [idx for idx, label in enumerate(labels) if label == 0]

        max_samples = min(len(positives), len(negatives))
        test_samples = int(max_samples * self.test_size)
        train_samples = max_samples - test_samples

        # Get an equal number of randomly chosen negative and positive examples
        positives = np.random.choice(positives, max_samples, replace=False)
        negatives = np.random.choice(negatives, max_samples, replace=False)

        # Now we just gotta slice these into training and testing indices
        train_positives = positives[:train_samples]
        train_negatives = negatives[:train_samples]

        test_positives = positives[train_samples:]
        test_negatives = negatives[train_samples:]

        train_idxs = np.concatenate((train_positives,train_negatives))
        shuffle(train_idxs)

        test_idxs = np.concatenate((test_positives,test_negatives))
        shuffle(test_idxs)

        return np.array(train_idxs).astype('int'), np.array(test_idxs).astype('int')





















