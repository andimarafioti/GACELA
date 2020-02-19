import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.colorize import colorize
from utils.consistencyComputer import consistency

import essentia
from essentia.standard import RhythmExtractor2013
import essentia.streaming as ess


class TensorboardSummarizer(object):
    def __init__(self, folderName, writeInterval):
        super(TensorboardSummarizer, self).__init__()
        self._summary_writer = SummaryWriter(folderName)
        self._writeInterval = writeInterval
        self._rhythm_extractor = RhythmExtractor2013(method="multifeature")
        self._tracked_scalars = {}

    def trackScalar(self, summaryName, scalar):
        if summaryName in self._tracked_scalars:
            self._tracked_scalars[summaryName] += scalar.detach().data.mean()
        else:
            self._tracked_scalars[summaryName] = scalar.detach().data.mean()

    def musicAnalysis(self, signal):
        bpm, beats, beats_confidence, _, beats_intervals = self._rhythm_extractor(np.single(signal))
        dissonance, inharmonicity, tuning_frequency = self.tonalAnalysis(signal)
        return beats_confidence, dissonance.mean(), dissonance.std(), \
               inharmonicity.mean(), inharmonicity.std(), tuning_frequency.mean(), tuning_frequency.std()

    def tonalAnalysis(self, signal):
        vectorinput = ess.VectorInput(np.single(signal))
        framecutter = ess.FrameCutter(frameSize=4096, hopSize=2048, silentFrames='noise')
        windowing = ess.Windowing(type='blackmanharris62')
        spectrum = ess.Spectrum()
        spectralpeaks = ess.SpectralPeaks(orderBy='frequency',
                                          magnitudeThreshold=1e-5,
                                          minFrequency=20,
                                          maxFrequency=3500,
                                          maxPeaks=60)

        dissonance = ess.Dissonance()
        tuning_frequency = ess.TuningFrequency()
        inharmonicity = ess.Inharmonicity()

        # Use pool to store data
        pool = essentia.Pool()

        # Connect streaming algorithms
        vectorinput.data >> framecutter.signal
        framecutter.frame >> windowing.frame >> spectrum.frame
        spectrum.spectrum >> spectralpeaks.spectrum
        spectralpeaks.magnitudes >> dissonance.magnitudes
        spectralpeaks.frequencies >> dissonance.frequencies
        spectralpeaks.magnitudes >> tuning_frequency.magnitudes
        spectralpeaks.frequencies >> tuning_frequency.frequencies
        spectralpeaks.magnitudes >> inharmonicity.magnitudes
        spectralpeaks.frequencies >> inharmonicity.frequencies

        dissonance.dissonance >> (pool, 'tonal.dissonance')
        inharmonicity.inharmonicity >> (pool, 'tonal.inharmonicity')
        tuning_frequency.tuningFrequency >> (pool, 'tonal.tuningFrequency')
        tuning_frequency.tuningCents >> (pool, 'tonal.tuningCents')

        # Run streaming network
        essentia.run(vectorinput)

        return pool['tonal.dissonance'], pool['tonal.inharmonicity'], pool['tonal.tuningFrequency']

    def writeSummary(self, batch_idx, real_spectrograms, generated_spectrograms, fake_spectrograms, fake_sounds, real_sounds, sampling_rate):
        for summaryName in self._tracked_scalars:
            self._summary_writer.add_scalar(summaryName, self._tracked_scalars[summaryName]/self._writeInterval,
                                            global_step=batch_idx)
        self._tracked_scalars = {}

        music_analysis_fake_signal = np.zeros([7, len(fake_sounds)])
        music_analysis_real_signal = np.zeros([7, len(real_sounds)])
        for index, (fake, real) in enumerate(zip(fake_sounds, real_sounds)):
            music_analysis_fake_signal[:, index] = self.musicAnalysis(fake)
            music_analysis_real_signal[:, index] = self.musicAnalysis(real)

        self._summary_writer.add_scalar("MusicAnalysis/Real_beats_confidence",
                                        np.mean(music_analysis_real_signal[0]), global_step=batch_idx)
        self._summary_writer.add_scalar("MusicAnalysis/Real_dissonance_mean",
                                        np.mean(music_analysis_real_signal[1]), global_step=batch_idx)
        self._summary_writer.add_scalar("MusicAnalysis/Real_dissonance_std",
                                        np.mean(music_analysis_real_signal[2]), global_step=batch_idx)
        self._summary_writer.add_scalar("MusicAnalysis/Real_inharmonicity_mean",
                                        np.mean(music_analysis_real_signal[3]), global_step=batch_idx)
        self._summary_writer.add_scalar("MusicAnalysis/Real_inharmonicity_std",
                                        np.mean(music_analysis_real_signal[4]), global_step=batch_idx)
        self._summary_writer.add_scalar("MusicAnalysis/Real_tuning_frequency_mean",
                                        np.mean(music_analysis_real_signal[5]), global_step=batch_idx)
        self._summary_writer.add_scalar("MusicAnalysis/Real_tuning_frequency_std",
                                        np.mean(music_analysis_real_signal[6]), global_step=batch_idx)

        self._summary_writer.add_scalar("MusicAnalysis/Fake_beats_confidence",
                                        np.mean(music_analysis_fake_signal[0]), global_step=batch_idx)
        self._summary_writer.add_scalar("MusicAnalysis/Fake_dissonance_mean",
                                        np.mean(music_analysis_fake_signal[1]), global_step=batch_idx)
        self._summary_writer.add_scalar("MusicAnalysis/Fake_dissonance_std",
                                        np.mean(music_analysis_fake_signal[2]), global_step=batch_idx)
        self._summary_writer.add_scalar("MusicAnalysis/Fake_inharmonicity_mean",
                                        np.mean(music_analysis_fake_signal[3]), global_step=batch_idx)
        self._summary_writer.add_scalar("MusicAnalysis/Fake_inharmonicity_std",
                                        np.mean(music_analysis_fake_signal[4]), global_step=batch_idx)
        self._summary_writer.add_scalar("MusicAnalysis/Fake_tuning_frequency_mean",
                                        np.mean(music_analysis_fake_signal[5]), global_step=batch_idx)
        self._summary_writer.add_scalar("MusicAnalysis/Fake_tuning_frequency_std",
                                        np.mean(music_analysis_fake_signal[6]), global_step=batch_idx)

        real_c = consistency((real_spectrograms - 1) * 25)
        fake_c = consistency((generated_spectrograms - 1) * 25)

        mean_R_Con, std_R_Con = real_c.mean(), real_c.std()
        mean_F_Con, std_F_Con = fake_c.mean(), fake_c.std()

        self._summary_writer.add_scalar("Gen/Reg", torch.abs(mean_R_Con - mean_F_Con), global_step=batch_idx)
        self._summary_writer.add_scalar("Gen/F_Con", mean_F_Con, global_step=batch_idx)
        self._summary_writer.add_scalar("Gen/F_STD_Con", std_F_Con, global_step=batch_idx)
        self._summary_writer.add_scalar("Gen/R_Con", mean_R_Con, global_step=batch_idx)
        self._summary_writer.add_scalar("Gen/R_STD_Con", std_R_Con, global_step=batch_idx)
        self._summary_writer.add_scalar("Gen/STD_diff", torch.abs(std_F_Con - std_R_Con), global_step=batch_idx)

        for index in range(4):
            self._summary_writer.add_image("images/Real_Image/" + str(index), colorize(real_spectrograms[index]),
                                     global_step=batch_idx)
            self._summary_writer.add_image("images/Fake_Image/" + str(index), colorize(fake_spectrograms[index], -1, 1),
                                     global_step=batch_idx)
            self._summary_writer.add_audio('sounds/Gen/' + str(index), fake_sounds[index]/(np.abs(fake_sounds[index]).max()), global_step=batch_idx, sample_rate=sampling_rate)
            self._summary_writer.add_audio('sounds/Real/' + str(index), real_sounds[index]/(np.abs(real_sounds[index]).max()), global_step=batch_idx, sample_rate=sampling_rate)
