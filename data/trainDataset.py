import os
import glob
import numpy as np
import torch
import torch.utils.data as data

__author__ = 'Andres'


class TrainDataset(data.Dataset):
    def __init__(self, root, window_size, examples_per_file=8, blacklist_patterns=None):
        assert (isinstance(root, str))
        if blacklist_patterns is None:
            blacklist_patterns = []

        self.root = root
        self._window_size = window_size
        self.filenames = np.random.shuffle(glob.glob(os.path.join(root, "*.dat")))
        self._examples_per_file = examples_per_file

        for pattern in blacklist_patterns:
            self.filenames = self.blacklist(self.filenames, pattern)

    def blacklist(self, filenames, pattern):
        return [filename for filename in filenames if pattern not in filename]

    def __len__(self):
        return len(self.filenames) * 1000

    def __getitem__(self, index):
        name = self.filenames[index % len(self.filenames)]
        spectrogram = np.memmap(name, mode='r', dtype=np.float64).reshape([257, -1])

        starts = np.random.randint(0, spectrogram.shape[1]-self._window_size, self._examples_per_file)

        spectrograms = np.zeros([self._examples_per_file, 256, self._window_size], dtype=np.float64)

        for index, start in enumerate(starts):
            spectrograms[index] = spectrogram[:256, start:start+self._window_size]

        return spectrograms


if __name__ == '__main__':
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
