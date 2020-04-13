import numpy as np

from data.baseDataset import BaseDataset

__author__ = 'Andres'


class TrainDataset(BaseDataset):
    def _saveNewFile(self, name, audio, spectrogram):
        self._loaded_files[name] = [0, spectrogram]

    def _sliceAudio(self, audio):
        return audio[:int(0.8*audio.shape[0])]

    def __getitem__(self, unused_index):
        filename = self._selectFile()
        spectrogram = self._loaded_files[filename][1]

        starts = np.random.randint(0, spectrogram.shape[1] - self._window_size, self._examples_per_file)

        spectrograms = np.zeros([self._examples_per_file, self._audio_loader.windowLength()//2+1, self._window_size], dtype=np.float64)

        for index, start in enumerate(starts):
            spectrograms[index] = spectrogram[:, start:start + self._window_size]

        self._usedFilename(filename)

        return spectrograms[:, :-1]


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
