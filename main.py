from dataloader import WayneLoanApprovalLoader
from experiment import StratifiedExperiment, BinaryBalancedExperiment
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB as NB
from custom_models import BinaryDummyModel
from matplotlib import pyplot as plt

from custom_models import LoanPytorchModel

#Pulling in all data from 2007-2014
wayne_all = WayneLoanApprovalLoader(savename='wayne_all', csvfile='wayne_county_2007_2014.tsv')

# We have some data, now lets choose a model and some metrics, before putting them into experiment objects.
lr1 = LogisticRegression()
lr2 = LogisticRegression()

lrb1 = LogisticRegression(class_weight='balanced')
lrb2 = LogisticRegression(class_weight='balanced')

ada1 = AdaBoostClassifier()
ada2 = AdaBoostClassifier()

timemodels = [lr1, lr2]

criterion = accuracy_score # Thankfully this task has a pretty easy evaluation... you either get it right or wrong

# Getting temporally contiguous cuts of data, putting them into different experiments
data_time1 = wayne_all.get_dates([2007, 2008, 2009, 2010])
expmt_time1 = StratifiedExperiment(timemodels[0], criterion, data_time1[:, :-1], data_time1[:, -1], test_size=0.8)

data_time2 = wayne_all.get_dates([2011, 2012, 2013, 2014])
expmt_time2 = StratifiedExperiment(timemodels[1], criterion, data_time2[:, :-1], data_time2[:, -1], test_size=0.8)


def do_runs(input_experiment, runs):

    trainvals = []
    testvals = []

    for run in range(runs):
        trainval, testval = input_experiment.run()
        trainvals.append(trainval)
        testvals.append(testval)

    print("\nTraining Validation: %6.2f, Testing Validation: %6.2f" % (np.mean(trainvals), np.mean(testvals)))

def get_conmat(input_model, features, labels, runs):

    conmat = np.zeros([2, 2])
    for run in range(runs):
        conmat += confusion_matrix(labels, input_model.predict(features))

    conmat = conmat / np.sum(conmat)

    print("\n")
    print(conmat)

# do_runs(expmt_time1, 10)
# do_runs(expmt_time2, 10)
#
#
# # Plotting the results of the time experiment
#
# plt.figure(0)
# tickpos = np.arange(len(wayne_all.vector_headers))*2
# one = plt.barh(tickpos, timemodels[0].coef_[0], align='center', alpha=0.5, height=2)
# plt.yticks(tickpos, wayne_all.vector_headers)
#
# two = plt.barh(tickpos, timemodels[1].coef_[0], align='center', alpha=0.5, height=2)
#
# plt.legend((one, two), ('2007-2010', '2011-2014'))
# plt.xlabel("Model Weight")
# plt.tick_params(axis='both', which='major', labelsize=8, rotation=45)
#
# plt.show()

# Neural net experiment
nn_model = LoanPytorchModel(wayne_all.data[:, :-1].shape[1], 1, batch_size=4, epochs=5, layers=4)
nn_experiment = StratifiedExperiment(nn_model, criterion, wayne_all.data[:, :-1], wayne_all.data[:, -1],
                                             test_size=0.2)

print("\nNEURAL NETWORK PERFORMANCE:")
do_runs(nn_experiment, 1)


# Just doing an experiment over all time now that we've established the distributions the observations are drawn from
# are fairly similar
strat_model = LogisticRegression()
stratified_experiment = StratifiedExperiment(strat_model, criterion, wayne_all.data[:, :-1], wayne_all.data[:, -1],
                                             test_size=0.8)

# To investigate failures let's look at the confusion matrix from multiple runs as well.

#I'm going to make a meta-split in the data used to train and test models, so we can compare experiment types.

## Trying stratified
train_idx, test_idx = stratified_experiment.partition(wayne_all.data[:, :-1], wayne_all.data[:, -1])
stratified_experiment = StratifiedExperiment(strat_model,
                                             criterion, wayne_all.data[train_idx, :-1],
                                             wayne_all.data[train_idx, -1], test_size=0.8)

print("\nSTRATIFIED:")
do_runs(stratified_experiment, 10)
get_conmat(strat_model, wayne_all.data[test_idx, :-1], wayne_all.data[test_idx, -1], 10)


## Trying balanced partitioning
balanced_model = LogisticRegression()
balanced_experiment = BinaryBalancedExperiment(balanced_model, criterion,
                                               wayne_all.data[train_idx, :-1], wayne_all.data[train_idx, -1],
                                               test_size=0.8)

print("\nBALANCED:")
do_runs(balanced_experiment, 10)
get_conmat(balanced_model, wayne_all.data[test_idx, :-1], wayne_all.data[test_idx, -1], 10)


## Trying weight balancing
weight_balanced_model = LogisticRegression(class_weight='balanced')
weight_balanced_experiment = StratifiedExperiment(weight_balanced_model, criterion,
                                                  wayne_all.data[train_idx, :-1], wayne_all.data[train_idx, -1],
                                                  test_size=0.8)

print("\nWEIGHT BALANCED:")
do_runs(weight_balanced_experiment, 10)
get_conmat(weight_balanced_model, wayne_all.data[test_idx, :-1], wayne_all.data[test_idx, -1], 10)


## Going back and checking confusion matrices for balanced LRs in different times
timemodels = [LogisticRegression(class_weight='balanced'), LogisticRegression(class_weight='balanced')]

expmt_time1 = StratifiedExperiment(timemodels[0], criterion, data_time1[:, :-1], data_time1[:, -1], test_size=0.8)
expmt_time2 = StratifiedExperiment(timemodels[1], criterion, data_time2[:, :-1], data_time2[:, -1], test_size=0.8)

# Quick model training
do_runs(expmt_time1, 1)
do_runs(expmt_time2, 1)

# Predictions of time models on first timeblock
time1_test_features = data_time1[expmt_time1.last_test_idx, :-1]
time1_test_labels = data_time1[expmt_time1.last_test_idx, -1]

print("\n2007-2010-trained classification behavior on 2007-2010 data")
get_conmat(timemodels[0], time1_test_features, time1_test_labels, 10)

print("\n2011-2014-trained classification behavior on 2007-2010 data")
get_conmat(timemodels[1], time1_test_features, time1_test_labels, 10)

# Predictions of time models on second timeblock
time2_test_features = data_time2[expmt_time2.last_test_idx, :-1]
time2_test_labels = data_time2[expmt_time2.last_test_idx, -1]

print("\n2007-2010-trained classification behavior on 2011-2014 data")
get_conmat(timemodels[0], time2_test_features, time2_test_labels, 10)

print("\n2011-2014-trained classification behavior on 2007-2010 data")
get_conmat(timemodels[1], time2_test_features, time2_test_labels, 10)
