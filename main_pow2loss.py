import torch
from torch.utils.tensorboard import SummaryWriter

from data.trainDataset import TrainDataset
from trainer_pow2loss import train
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


signal_split = [192, 128, 192]
md = 32

params_discriminator = dict()
params_discriminator['stride'] = [2,2,2,2,2]
params_discriminator['nfilter'] = [md, 2*md, 4*md, 8*md, 16*md]
params_discriminator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
params_discriminator['full'] = []
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True
params_discriminator['data_size'] = 2
params_discriminator['apply_phaseshuffle'] = True
params_discriminator['spectral_norm'] = True

params_generator = dict()
params_generator['stride'] = [2, 2, 2, 2, 2]
params_generator['latent_dim'] = 100
params_generator['nfilter'] = [8*md, 4*md, 2*md, md, 1]
params_generator['shape'] = [[4, 4], [4, 4], [8, 8], [8, 8], [8, 8]]
params_generator['padding'] = [[1, 1], [1, 1], [3, 3], [3, 3], [3, 3]]
params_generator['full'] = 256*md
params_generator['summary'] = True
params_generator['data_size'] = 2
params_generator['spectral_norm'] = True
params_generator['in_conv_shape'] = [8, 4]
params_generator['borders'] = dict()
params_generator['borders']['nfilter'] = [md, 2*md, md, md/2]
params_generator['borders']['shape'] = [[5, 5],[5, 5],[5, 5],[5, 5]]
params_generator['borders']['stride'] = [2, 2, 3, 4]
params_generator['borders']['data_size'] = 2
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
params_optimization['batch_size'] = 64*2
params_discriminator['batch_size'] = 64*2

params_optimization['epoch'] = 600
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
params['net'] = dict() # All the parameters for the model
params['net']['generator'] = params_generator
params['net']['discriminator'] = params_discriminator
params['net']['prior_distribution'] = 'gaussian'
params['net']['shape'] = [1, 256, 128*4] # Shape of the image
params['net']['inpainting']=dict()
params['net']['inpainting']['split']=signal_split
params['net']['gamma_gp'] = 10 # Gradient penalty
# params['net']['fs'] = 16000//downscale
params['net']['loss_type'] ='wasserstein'

params['optimization'] = params_optimization
params['summary_every'] = 250 # Tensorboard summaries every ** iterations
params['print_every'] = 50 # Console summaries every ** iterations
params['save_every'] = 1000 # Save the model every ** iterations
# params['summary_dir'] = os.path.join(global_path, name +'_summary/')
# params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')
params['Nstats'] = 500

args = dict()
args['generator'] = params_generator
args['discriminator'] = params_discriminator
args['borderEncoder'] = params_generator['borders']
args['discriminator_in_shape'] = [1, 256, 128]
args['generator_input'] = 2*6*4*2*8
args['optimizer'] = params_optimization
args['split'] = signal_split
args['log_interval'] = 50
args['spectrogram_shape'] = params['net']['shape']
args['gamma_gp'] = params['net']['gamma_gp']
args['tensorboard_interval'] = 250
args['save_path'] = '../saved_results/'
args['experiment_name'] = 'pytorch_nc1_pow2loss_2'
args['save_interval'] = 1000


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

examples_per_file = 16
trainDataset = TrainDataset("../data/Maestro_spectrograms_mep/", window_size=512, examples_per_file=examples_per_file)

train_loader = torch.utils.data.DataLoader(trainDataset,
    batch_size=args['optimizer']['batch_size']//examples_per_file, shuffle=True,
                                           num_workers=4, drop_last=True)

summary_writer = SummaryWriter(args['save_path'] + args['experiment_name'] + '_summary')
start_at = 0

for epoch in range(10):
    start_at, can_restart = train(args, device, train_loader, epoch, summary_writer, start_at)
    if not can_restart:
        break