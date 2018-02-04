from dataloader import WayneLoanApprovalLoader
from experiment import StratifiedExperiment
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier

from matplotlib import pyplot as plt

wayne_all = WayneLoanApprovalLoader(savename='wayne_all', csvfile='wayne_county_2007_2014.tsv')

# We have some data, now lets choose a model and some metrics.
lr1 = LogisticRegression()
lr2 = LogisticRegression()

ada1 = AdaBoostClassifier()
ada2 = AdaBoostClassifier()

timemodels = [lr1, lr2]

criterion = accuracy_score # Thankfully this task has a pretty easy evaluation... you either get it right or wrong

data_time1 = wayne_all.get_dates([2007, 2008, 2009, 2010])
expmt_time1 = StratifiedExperiment(timemodels[0], criterion, data_time1[:, :-1], data_time1[:, -1], test_size=0.1)

data_time2 = wayne_all.get_dates([2011, 2012, 2013, 2014])
expmt_time2 = StratifiedExperiment(timemodels[1], criterion, data_time2[:, :-1], data_time2[:, -1], test_size=0.1)


def do_runs(experiment, runs):

    trainvals = []
    testvals = []

    for run in range(runs):
        trainval, testval = experiment.run()
        trainvals.append(trainval)
        testvals.append(testval)

    print("\nTraining Validation: %6.2f, Testing Validation: %6.2f" % (np.mean(trainvals), np.mean(testvals)))


# do_runs(expmt_time1, 10)
# do_runs(expmt_time2, 10)
#
#
# # Plotting the results of the time experiment
#
# plt.figure(0)
# tickpos = np.arange(len(wayne_all.vector_headers))*2
# one = plt.barh(tickpos, lr1.coef_[0], align='center', alpha=0.5, height=2)
# plt.yticks(tickpos, wayne_all.vector_headers)
#
# two = plt.barh(tickpos, lr2.coef_[0], align='center', alpha=0.5, height=2)
#
# plt.legend((one, two), ('2007-2010', '2011-2014'))
# plt.xlabel("Model Weight")
# plt.tick_params(axis='both', which='major', labelsize=8, rotation=45)
#
# plt.show()

# Just doing an experiment over all time now that we've established the distributions the observations are drawn from
# are fairly similar
model = LogisticRegression()
experiment = StratifiedExperiment(model, criterion, wayne_all.data[:, :-1], wayne_all.data[:, -1], test_size=0.9)

#Do some runs to get raw accuracies
do_runs(experiment, 10)

# To investigate failures let's look at the confusion matrix from multiple runs as well.
conmat = np.zeros([2,2])
for run in range(10):
    train_idx, test_idx = experiment.partition(wayne_all.data[:, :-1], wayne_all.data[:, -1])
    model.fit(wayne_all.data[train_idx, :-1], wayne_all.data[train_idx, -1])
    conmat += confusion_matrix(wayne_all.data[test_idx, -1], model.predict(wayne_all.data[test_idx, :-1]))

conmat = conmat / np.sum(conmat)

print("\n")
print(conmat)