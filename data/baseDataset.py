import numpy as np
from pathlib import Path
import torch
import torch.utils.data as data

from utils.worker import Worker
__author__ = 'Andres'


class BaseDataset(data.Dataset):
    def __init__(self, root, window_size, audio_loader, examples_per_file=8, blacklist_patterns=None, loaded_files_buffer=10,
                 file_usages=10):
        assert (isinstance(root, str))
        if blacklist_patterns is None:
            blacklist_patterns = []

        self.root = root
        self._window_size = window_size
        self._audio_loader = audio_loader
        self._loaded_files_buffer = loaded_files_buffer
        self._file_usages = file_usages

        self._index = 0
        self.filenames = [filename for filename in Path(root).rglob('*.wav')]
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
        self._index += 1
        audio = self._audio_loader.loadSound(name)
        audio = self._sliceAudio(audio)
        spectrogram = self._audio_loader.computeSpectrogram(audio)
        if spectrogram.shape[1] <= self._window_size:
            self._loadNewFile()
            return
        self._saveNewFile(name, audio, spectrogram)

    def _saveNewFile(self, name, audio, spectrogram):
        raise NotImplementedError("Subclass responsibility")

    def _usedFilename(self, filename):
        self._loaded_files[filename][0] = self._loaded_files[filename][0] + 1
        if self._loaded_files[filename][0] >= self._file_usages:
            del self._loaded_files[filename]
            Worker.call(self._loadNewFile).asDaemon.start()

    def _selectFile(self):
        loaded_files = list(self._loaded_files.keys())
        while len(list(self._loaded_files.keys())) == 0:
            self._loadNewFile()
            Worker.call(self._populateLoadedFiles).asDaemon.start()
            loaded_files = list(self._loaded_files.keys())

        return np.random.choice(loaded_files)

    def __len__(self):
        return len(self.filenames) * 1000

    def _sliceAudio(self, audio):
        raise NotImplementedError("Subclass responsibility")

    def __getitem__(self, unused_index):
        raise NotImplementedError("Subclass responsibility")


