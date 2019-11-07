from data.baseDataset import BaseDataset

__author__ = 'Andres'


class ValidDataset(BaseDataset):
    def _sliceData(self, data):
        return data[:, int(0.9*data.shape[1]):]
