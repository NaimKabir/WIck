from dataloader import WayneLoanApprovalLoader
from experiment import StratifiedExperiment
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble.forest import RandomForestClassifier

wayne_2007_2010 = WayneLoanApprovalLoader('wayne_county_2007_2010.tsv')
wayne_2011_2014 = WayneLoanApprovalLoader('wayne_county_2011_2014.tsv')

# We have some data, now lets choose a model and some metrics.
model_1 = RandomForestClassifier()
model_2 = RandomForestClassifier()
criterion = accuracy_score # Thankfully this task has a pretty easy evaluation... you either get it right or wrong

expmt_2007_2010 = StratifiedExperiment(model_1, criterion, wayne_2007_2010.data[:, :-1], wayne_2007_2010.data[:, -1])
expmt_2011_2014 = StratifiedExperiment(model_2, criterion, wayne_2011_2014.data[:, :-1], wayne_2011_2014.data[:, -1])

trainvals = []
testvals = []
runs = 10

for run in range(runs):
    trainval, testval = expmt_2007_2010.run()
    trainvals.append(trainval)
    testvals.append(testval)

print("Training Validation: %6.2f, Testing Validation: %6.2f" % (np.mean(trainvals), np.mean(testvals)))

trainvals = []
testvals = []
runs = 10

for run in range(runs):
    trainval, testval = expmt_2011_2014.run()
    trainvals.append(trainval)
    testvals.append(testval)

print("Training Validation: %6.2f, Testing Validation: %6.2f" % (np.mean(trainvals), np.mean(testvals)))