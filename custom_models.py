import numpy as np

class Model(object):

    def __init__(self): raise NotImplementedError

    def fit(self, features, labels): raise NotImplementedError

    def predict(self, features): raise NotImplementedError


class BinaryDummyModel(Model):

    def __init__(self):

        pass

    def fit(self, features, labels):
        pass

    def predict(self, features):

        return np.ones([features.shape[0]])