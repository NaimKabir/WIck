from sklearn.model_selection import StratifiedShuffleSplit


# Abstract class for experiments on data
class Experiment(object):

    def __init__(self, model, criterion, features, labels):

        self.features = features
        self.labels = labels
        self.model = model
        self.criterion = criterion

    # Split up my data into training and testing
    # Can stick shuffles, stratification, etc. in here.
    def partition(self, features, labels): raise NotImplementedError

    # Run the experiment and get back a  a training and testing validation, preferably averaged across folds
    def run(self): raise NotImplementedError

# Class for performing a stratified shuffle for a single experimental run
class StratifiedExperiment(Experiment):

    def __init__(self, model, criterion, features, labels):

        super(StratifiedExperiment, self).__init__(model, criterion, features, labels)

    # In this partition I'm doing a stratified shuffle
    def partition(self, features, labels):

        shuf = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
        splits = shuf.split(features, labels)

        for train_idx, test_idx in splits:
            return train_idx, test_idx

    def run(self):

        train_idx, test_idx = self.partition(self.features, self.labels)

        train_features, train_labels = self.features[train_idx, :], self.labels[train_idx]
        test_features, test_labels = self.features[test_idx, :], self.labels[test_idx]

        self.model.fit(train_features, train_labels)

        train_metric = self.criterion(self.model.predict(train_features), train_labels)
        test_metric = self.criterion(self.model.predict(test_features), test_labels)

        return train_metric, test_metric










