import numpy as np
from data.baseDataset import BaseDataset

__author__ = 'Andres'


class ValidDataset(BaseDataset):
    def _sliceAudio(self, audio):
        return audio[int(0.8*audio.shape[0]):]

    def _saveNewFile(self, name, audio, spectrogram):
        self._loaded_files[name] = [0, spectrogram, audio]
        self._index += 1

    def __getitem__(self, unused_index):
        filename = self._selectFile()
        spectrogram, audio = self._loaded_files[filename][1], self._loaded_files[filename][2]

        starts = np.random.randint(0, spectrogram.shape[1] - self._window_size, self._examples_per_file)

        spectrograms = np.zeros([self._examples_per_file, self._audio_loader.windowLength()//2+1, self._window_size], dtype=np.float64)
        audio_length = self._window_size*self._audio_loader.hopSize()
        audios = np.zeros([self._examples_per_file, audio_length])

        for index, start in enumerate(starts):
            spectrograms[index] = spectrogram[:, start:start + self._window_size]
            audio_start = np.min([start*self._audio_loader.hopSize(), audio.shape[0]-audio_length])
            audios[index] = audio[audio_start:audio_start+audio_length]
        self._usedFilename(filename)

        return spectrograms[:, :-1], audios
