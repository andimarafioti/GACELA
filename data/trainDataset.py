from data.baseDataset import BaseDataset

__author__ = 'Andres'


class TrainDataset(BaseDataset):
    def _sliceData(self, data):
        return data[:, :int(0.9*data.shape[1])]

if __name__ == '__main__':
    import torch
    examples_per_file = 16
    dataset = TrainDataset("", window_size=512, examples_per_file=examples_per_file)
    print(dataset[1].shape)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=128 // 16,
                                               shuffle=True)

    for _ in train_loader:
        print(_.shape)
        _= _.view(128, *[1, 256, 128*4])

        print(_.shape)
