from dataloader import WayneLoanApprovalLoader
from experiment import StratifiedExperiment
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.bagging import BaggingClassifier

wayne_all = WayneLoanApprovalLoader(savename='wayne_all', csvfile='wayne_county_2007_2014.tsv')

# We have some data, now lets choose a model and some metrics.
lr1 = LogisticRegression()
lr2 = LogisticRegression()


criterion = accuracy_score # Thankfully this task has a pretty easy evaluation... you either get it right or wrong

data_2007_2010 = wayne_all.get_dates([2007, 2008, 2009, 2010])
expmt_2007_2010 = StratifiedExperiment(lr1, criterion, data_2007_2010[:, :-1], data_2007_2010[:, -1])

data_2011_2014 = wayne_all.get_dates([2011, 2012, 2013, 2014])
expmt_2011_2014 = StratifiedExperiment(lr2, criterion, data_2011_2014[:, :-1], data_2011_2014[:, -1])


def do_runs(experiment, runs):

    trainvals = []
    testvals = []

    for run in range(runs):
        trainval, testval = experiment.run()
        trainvals.append(trainval)
        testvals.append(testval)

    print("Training Validation: %6.2f, Testing Validation: %6.2f" % (np.mean(trainvals), np.mean(testvals)))





#
# trainvals = []
# testvals = []
# runs = 10
#
# for run in range(runs):
#     trainval, testval = expmt_2007_2010.run()
#     trainvals.append(trainval)
#     testvals.append(testval)
#
# print("Training Validation: %6.2f, Testing Validation: %6.2f" % (np.mean(trainvals), np.mean(testvals)))
#
# trainvals = []
# testvals = []
# runs = 10
#
# for run in range(runs):
#     trainval, testval = expmt_2011_2014.run()
#     trainvals.append(trainval)
#     testvals.append(testval)
#
# print("Training Validation: %6.2f, Testing Validation: %6.2f" % (np.mean(trainvals), np.mean(testvals)))