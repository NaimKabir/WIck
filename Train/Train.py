import torch
from torch.utils.data import Dataset, DataLoader

class Train():

    def __init__(self, dataset):

        self.loader = dataset
