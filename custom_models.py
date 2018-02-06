import numpy as np
import torch as th
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim import Adadelta, SGD

# Class to make sure all model objects jive well with the rest of the code body
class Model(object):

    def __init__(self): raise NotImplementedError

    def fit(self, features, labels): raise NotImplementedError

    def predict(self, features): raise NotImplementedError


# Test model helpful in debugging
class BinaryDummyModel(Model):

    def __init__(self):
        super(BinaryDummyModel, self).__init__()
        pass

    def fit(self, features, labels):
        pass

    def predict(self, features):
        return np.ones([features.shape[0]])

# General model class for pytorch models.
class PytorchModel(Model):

    def __init__(self, batch_size, early_stopping_threshold, epochs):

        self.batch_size = batch_size
        self.early_stopping_threshold = early_stopping_threshold
        self.epochs = epochs

        self.net = None
        self.loss = None
        self.optimizer = None

    # A neural net training method that just stops whenever the diffs in losses get below an arbitrary threshold.
    # I can also make it epoch based... maybe. #TODO
    def fit(self, features, labels):

        features, labels = th.FloatTensor(features), th.FloatTensor(labels)

        dataloader = DataLoader(RAMPytorchDataset(features, labels), batch_size=self.batch_size, shuffle=True)

        # This is the training loop.
        # Keep track of losses and stop when the time smoothed average falls below threshold
        losses = []

        for epoch in range(self.epochs):

            for feature, label in dataloader:

                feature, label = Variable(feature), Variable(label)

                # Making a prediction and calculating a loss
                prediction = self.net.forward(feature)
                loss = self.loss(prediction, label)

                losses.append(loss.data)
                if len(losses) > 100:
                    losses.pop(0)

                # if np.abs(np.mean(np.diff(losses))) < self.early_stopping_threshold:
                #     # Saving model just in case it took a ridiculous amount of time for it to get good.
                #     th.save(self.net, "nn.pt")
                #     break

                # Back-propagating and and then applying the gradients to the neural network weights.
                loss.backward()
                self.optimizer.step()

            print("Completed epoch: %d with Loss: %6.2f" % (epoch, np.mean(losses)))

        th.save(self.net, "nn.pt")



    def predict(self, features):

        features = Variable(th.FloatTensor(features))

        # Going to force it to round up or down.
        return th.round(self.net.forward(features).data).numpy()

# This is the tiny class I'm actually using to predict things in the task
class LoanPytorchModel(PytorchModel):

    def __init__(self, input_dim, output_dim, batch_size, epochs, layers=2):

        early_stopping_threshold = 0.01
        super(LoanPytorchModel, self).__init__(batch_size, early_stopping_threshold, epochs)

        self.net = LoanNet(input_dim, output_dim, layers)
        self.loss = nn.BCELoss() # Because we're training for a binary classification.
        self.optimizer = SGD(self.net.parameters(), lr=0.1)

# Actual neural  net model I can plug into one of my classes above to help with my problem.
class LoanNet(nn.Module):

    # Takes in input and output dimensions desires, as well as the number of intermediate layers.
    # Nothing fancy for adding layers, they're all the same dimensionality, though I can play around with this later
    # TODO
    def __init__(self, input_dim, output_dim, layers):
        super(LoanNet, self).__init__()

        input_dim = int(input_dim)
        half_dim = int(input_dim/2)
        output_dim = int(output_dim)

        layer_dict = OrderedDict()
        layer_dict['linear0'] = nn.Linear(input_dim, half_dim)
        layer_dict['relu0'] = nn.ReLU()

        for layer in range(layers):
            layer_dict['linear' + str(layer + 1)] = nn.Linear(half_dim, half_dim)
            layer_dict['nonlinear' + str(layer + 1)] = nn.Sigmoid()

        layer_dict['final'] = nn.Linear(half_dim, output_dim)
        layer_dict['nonlinear'] = nn.Sigmoid() # Final layer because I want to output a probability.

        self.net = nn.Sequential(layer_dict)

    # Rather simple forward method... just pass it through the net.
    def forward(self, features):

        return self.net.forward(features)

# Building a small dataset class so I can make a dataloader with it for NN training
class RAMPytorchDataset(Dataset):

    # Get torch tensors as features and labels.
    # Assume they are aligned.
    def __init__(self, features, labels):

        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.size()[0]

    def __getitem__(self, item):

        return self.features[item, :], th.FloatTensor([self.labels[item]])