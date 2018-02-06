from custom_models import LoanPytorchModel
from dataloader import WayneLoanApprovalLoader
from sklearn.metrics import accuracy_score
from experiment import StratifiedExperiment


## quick test to debug pytorch stuff...
criterion = accuracy_score

test = WayneLoanApprovalLoader(savename='test', csvfile='test.tsv')

nn_model = LoanPytorchModel(test.data[:, :-1].shape[1], 1, batch_size=4)

nn_experiment = StratifiedExperiment(nn_model, criterion, test.data[:, :-1], test.data[:, -1], test_size=0.2)

nn_experiment.run()