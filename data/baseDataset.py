import os
import glob
import numpy as np
import torch
import torch.utils.data as data

from utils.worker import Worker
__author__ = 'Andres'


class BaseDataset(data.Dataset):
    def __init__(self, root, window_size, examples_per_file=8, blacklist_patterns=None, loaded_files_buffer=10,
                 file_usages=10):
        assert (isinstance(root, str))
        if blacklist_patterns is None:
            blacklist_patterns = []

        self.root = root
        self._window_size = window_size
        self._loaded_files_buffer = loaded_files_buffer
        self._file_usages = file_usages

        self._index = 0
        self.filenames = glob.glob(os.path.join(root, "*.dat"))
        for pattern in blacklist_patterns:
            self.filenames = self.blacklist(self.filenames, pattern)

        self._loaded_files = dict()
        Worker.call(self._populateLoadedFiles).asDaemon.start()

        np.random.shuffle(self.filenames)
        self._examples_per_file = examples_per_file

    def blacklist(self, filenames, pattern):
        return [filename for filename in filenames if pattern not in filename]

    def _populateLoadedFiles(self):
        while len(self._loaded_files) < self._loaded_files_buffer:
            self._loadNewFile()

    def _loadNewFile(self):
        name = self.filenames[self._index % len(self.filenames)]
        spectrogram = np.memmap(name, mode='r', dtype=np.float64).reshape([257, -1])
        self._loaded_files[name] = [0, spectrogram]
        self._index += 1

    def _usedFilename(self, filename):
        self._loaded_files[filename][0] = self._loaded_files[filename][0] + 1
        if self._loaded_files[filename][0] > self._file_usages:
            del self._loaded_files[filename]
            Worker.call(self._loadNewFile).asDaemon.start()

    def __len__(self):
        return len(self.filenames) * 1000

    def _sliceData(self, data):
        raise NotImplementedError("Subclass responsibility")

    def __getitem__(self, unused_index):
        loaded_files = list(self._loaded_files.keys())
        if len(list(self._loaded_files.keys())) == 0:
            self._loadNewFile()
            Worker.call(self._populateLoadedFiles).asDaemon.start()
            loaded_files = list(self._loaded_files.keys())

        filename = np.random.choice(loaded_files)
        spectrogram = self._loaded_files[filename][1]
        spectrogram = self._sliceData(spectrogram)

        starts = np.random.randint(0, spectrogram.shape[1] - self._window_size, self._examples_per_file)

        spectrograms = np.zeros([self._examples_per_file, 256, self._window_size], dtype=np.float64)

        for index, start in enumerate(starts):
            spectrograms[index] = spectrogram[:256, start:start + self._window_size]

        self._usedFilename(filename)

        return spectrograms


