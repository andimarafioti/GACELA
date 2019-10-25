import os
import glob
import numpy as np
import scipy.io.wavfile
import torch.utils.data as data

__author__ = 'Andres'


class TrainDataset(data.Dataset):
    def __init__(self, root, window_size, examples_per_file=8, blacklist_patterns=None):
        assert (isinstance(root, str))
        if blacklist_patterns is None:
            blacklist_patterns = []

        self.root = root
        self._window_size = window_size
        self.filenames = glob.glob(os.path.join(root, "*.mat"))
        self._examples_per_file = examples_per_file

        for pattern in blacklist_patterns:
            self.filenames = self.blacklist(self.filenames, pattern)

    def blacklist(self, filenames, pattern):
        return [filename for filename in filenames if pattern not in filename]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        name = self.filenames[index]
        spectrogram = scipy.io.loadmat(name)['logspec']  # load audio

        starts = np.random.randint(0, spectrogram.shape[1]-self._window_size, self._examples_per_file)

        spectrograms = np.zeros([self._examples_per_file, 256, self._window_size], dtype=np.float64)

        for index, start in enumerate(starts):
            spectrograms[index] = spectrogram[:256, start:start+self._window_size]

        return spectrograms


if __name__ == '__main__':
    dataset = TrainDataset('', 128, 16)
    print(dataset[1].shape)
