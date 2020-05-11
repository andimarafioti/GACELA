import sys
sys.path.insert(0, '../')

import torch

from data.audioLoader import AudioLoader
from data.trainDataset import TrainDataset
from ganSystem import GANSystem
import logging

# logging.getLogger().setLevel(logging.DEBUG)  # set root logger to debug

"""Just so logging works..."""
formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)
"""Just so logging works..."""

__author__ = 'Andres'

signal_split = [448, 128, 448]
md = 32

params_stft_discriminator = dict()
params_stft_discriminator['stride'] = [2, 2, 2, 2, 2]
params_stft_discriminator['nfilter'] = [md, 2 * md, 4 * md, 8 * md, 16 * md]
params_stft_discriminator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
params_stft_discriminator['data_size'] = 2

params_mel_discriminator = dict()
params_mel_discriminator['stride'] = [2, 2, 2, 2, 2]
params_mel_discriminator['nfilter'] = [md//4, 2 * md//4, 4 * md//4, 8 * md//4, 16 * md//4]
params_mel_discriminator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
params_mel_discriminator['data_size'] = 2

params_generator = dict()
params_generator['stride'] = [2, 2, 2, 2, 2]
params_generator['nfilter'] = [8 * md, 4 * md, 2 * md, md, 1]
params_generator['shape'] = [[4, 4], [4, 4], [8, 8], [8, 8], [8, 8]]
params_generator['padding'] = [[1, 1], [1, 1], [3, 3], [3, 3], [3, 3]]
params_generator['residual_blocks'] = 2

params_generator['full'] = 256 * md
params_generator['summary'] = True
params_generator['data_size'] = 2
params_generator['in_conv_shape'] = [16, 4]
params_generator['borders'] = dict()
params_generator['borders']['nfilter'] = [md, 2 * md, md, md / 2]
params_generator['borders']['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5]]
params_generator['borders']['stride'] = [2, 2, 2, 2]
params_generator['borders']['data_size'] = 2
params_generator['borders']['border_scale'] = 1
# This does not work because of flipping, border 2 need to be flipped tf.reverse(l, axis=[1]), ask Nathanael
params_generator['borders']['width_full'] = None

# Optimization parameters inspired from 'Self-Attention Generative Adversarial Networks'
# - Spectral normalization GEN DISC
# - Batch norm GEN
# - TTUR ('GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium')
# - ADAM  beta1=0 beta2=0.9, disc lr 0.0004, gen lr 0.0001
# - Hinge loss
# Parameters are similar to the ones in those papers...
# - 'PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION'
# - 'LARGE SCALE GAN TRAINING FOR HIGH FIDELITY NATURAL IMAGE SYNTHESIS'
# - 'CGANS WITH PROJECTION DISCRIMINATOR'

params_optimization = dict()
params_optimization['batch_size'] = 64
params_stft_discriminator['batch_size'] = 64
params_mel_discriminator['batch_size'] = 64

params_optimization['n_critic'] = 1
params_optimization['generator'] = dict()
params_optimization['generator']['optimizer'] = 'adam'
params_optimization['generator']['kwargs'] = [0.5, 0.9]
params_optimization['generator']['learning_rate'] = 1e-4
params_optimization['discriminator'] = dict()
params_optimization['discriminator']['optimizer'] = 'adam'
params_optimization['discriminator']['kwargs'] = [0.5, 0.9]
params_optimization['discriminator']['learning_rate'] = 1e-4

# all parameters
params = dict()
params['net'] = dict()  # All the parameters for the model
params['net']['generator'] = params_generator
params['net']['stft_discriminator'] = params_stft_discriminator
params['net']['mel_discriminator'] = params_mel_discriminator
params['net']['prior_distribution'] = 'gaussian'
params['net']['shape'] = [1, 512, 1024]  # Shape of the image
params['net']['inpainting'] = dict()
params['net']['inpainting']['split'] = signal_split
params['net']['gamma_gp'] = 10  # Gradient penalty
# params['net']['fs'] = 16000//downscale
params['net']['loss_type'] = 'wasserstein'

params['optimization'] = params_optimization
params['summary_every'] = 250  # Tensorboard summaries every ** iterations
params['print_every'] = 50  # Console summaries every ** iterations
params['save_every'] = 1000  # Save the model every ** iterations
# params['summary_dir'] = os.path.join(global_path, name +'_summary/')
# params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')

args = dict()
args['generator'] = params_generator
args['stft_discriminator_count'] = 2
args['mel_discriminator_count'] = 2
args['stft_discriminator'] = params_stft_discriminator
args['mel_discriminator'] = params_mel_discriminator
args['borderEncoder'] = params_generator['borders']
args['stft_discriminator_in_shape'] = [1, 512, 128]
args['mel_discriminator_in_shape'] = [1, 80, 128]
args['mel_discriminator_start_powscale'] = 2
args['generator_input'] = 1260
args['optimizer'] = params_optimization
args['split'] = signal_split
args['log_interval'] = 100
args['spectrogram_shape'] = params['net']['shape']
args['gamma_gp'] = params['net']['gamma_gp']
args['tensorboard_interval'] = 500
args['save_path'] = '../saved_results/'
args['experiment_name'] = 'real_data_448_128_448'
args['save_interval'] = 10000

args['fft_length'] = 1024
args['fft_hop_size'] = 256
args['sampling_rate'] = 22050

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

examples_per_file = 32
audioLoader = AudioLoader(args['sampling_rate'], args['fft_length'], args['fft_hop_size'], 50)

dataFolder = "../../../../Datasets/maestro-v2.0.0/"

trainDataset = TrainDataset(dataFolder, window_size=1024, audio_loader=audioLoader, examples_per_file=examples_per_file,
                            loaded_files_buffer=20, file_usages=30)

train_loader = torch.utils.data.DataLoader(trainDataset,
                                           batch_size=args['optimizer']['batch_size'] // examples_per_file,
                                           shuffle=True,
                                           num_workers=4, drop_last=True)

start_at_step = 234757
start_at_epoch = 1

ganSystem = GANSystem(args)

for epoch in range(start_at_epoch, 10):
    start_at_step, can_restart = ganSystem.train(train_loader, epoch, start_at_step)
    if not can_restart:
        break
