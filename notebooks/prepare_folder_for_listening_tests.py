import librosa
from pydub import AudioSegment
import numpy as np
import os

def match_target_amplitude(sound, target_dBFS=-10):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

new_folder = 'listening_test'
os.mkdir(new_folder)

walked_os = list(os.walk('.'))  # We are going to add files here and we do not want to walk them too

for path, subdirs, files in walked_os:
     for name in files:
        filename = os.path.join(path, name)
        if not filename.endswith('.wav'):
            continue

        sound, fs = librosa.core.load(filename)
        resampled_sound = librosa.core.resample(sound, fs, 48000)

        shifted_sound = (resampled_sound * (2 ** 31 - 1)).astype(np.int32)

        sound = AudioSegment(shifted_sound.tobytes(),
        					 frame_rate=48000,
                             sample_width=4, #  4 bytes, so 32 bit sample
                             channels=1)     #  mono

        sound = sound.fade_in(500)
        sound = sound.fade_out(500)

        normalized_sound = match_target_amplitude(sound)

        input_path_aslist = filename.split('\\')
        output_path_aslist = [input_path_aslist[0], new_folder, *input_path_aslist[1:]]

        if not os.path.exists(os.path.join(*output_path_aslist[:-1])):
            os.mkdir(os.path.join(*output_path_aslist[:-1]))

        output_name = os.path.join(*output_path_aslist)
        normalized_sound.export(output_name, format='wav')
