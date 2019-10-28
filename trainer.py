import torch
from torch import nn

from model.borderEncoder import BorderEncoder
from model.discriminator import Discriminator
from model.generator import Generator

import tqdm

from utils.wassersteinGradientPenalty import calc_gradient_penalty

__author__ = 'Andres'


def train(args, device, train_loader, epoch):
    discriminators = nn.ModuleList(
            [Discriminator(args['discriminator'], args['discriminator_in_shape']) for _ in range(3)]
        )

    left_border_encoder = BorderEncoder(args['borderEncoder'])
    right_border_encoder = BorderEncoder(args['borderEncoder'])

    generator = Generator(args['generator'], args['generator_input'])

    optim_g = torch.optim.Adam(list(generator.parameters()) + list(left_border_encoder.parameters()) +
                               list(right_border_encoder.parameters()),
                               lr=args['optimizer']['generator']['learning_rate'],
                               betas=(args['optimizer']['generator']['kwargs']))
    optim_d = torch.optim.Adam(discriminators.parameters(),
                               lr=args['optimizer']['discriminator']['learning_rate'],
                               betas=(args['optimizer']['discriminator']['kwargs']))

    # try:
    generator.train()
    discriminators.train()
    print('try')

    # train_loader = tqdm.tqdm(train_loader)
    for batch_idx, data in enumerate(train_loader):
        print('training!')
        print(batch_idx)

        data = data.to(device).float()
        data = data.view(args['optimizer']['batch_size'], *args['spectrogram_shape'])

        optim_g.zero_grad()
        optim_d.zero_grad()

        real_spectrograms = data[::2]

        fake_left_borders = data[1::2, :, :, :args['split'][0]]
        fake_right_borders = data[1::2, :, :, args['split'][0]+args['split'][1]:]

        encoded_left_border = left_border_encoder(fake_left_borders)
        print('border')
        print(encoded_left_border.shape)
        encoded_right_border = right_border_encoder(fake_right_borders)
        generated_spectrograms = generator(torch.cat((encoded_left_border, encoded_right_border), 1))

        print(fake_left_borders.shape)
        print(generated_spectrograms.shape)
        print(fake_right_borders.shape)

        fake_spectrograms = torch.cat((fake_left_borders, generated_spectrograms, fake_right_borders), 3)
        print(fake_spectrograms.shape)
        print('generator')
        d_loss_f = 0
        d_loss_r = 0
        d_loss_gp = 0

        for index, discriminator in enumerate(discriminators):
            scale = 2**index
            time = args['spectrogram_shape'][2]
            start = int((time - (time // 4) * scale) / 2)
            end = time - start
            x_fake = fake_spectrograms[:, :, :, start:end:scale]
            x_real = real_spectrograms[:, :, :, start:end:scale]
            print(x_fake.shape)
            print(x_real.shape)
            print('aaaaaaaa')

            d_loss_f += torch.mean(discriminator(x_fake))
            d_loss_r += torch.mean(discriminator(x_real))

            d_loss_gp += torch.mean(calc_gradient_penalty(discriminator, x_real, x_fake, args['gamma_gp']))

        disc_loss = -(d_loss_r - d_loss_f) + d_loss_gp
        gen_loss = d_loss_f

        gen_loss.backward(retain_graph=True)
        optim_g.step()

        for _ in range(args['optimizer']['n_critic']-1):
            optim_d.zero_grad()
            disc_loss.backward(retain_graph=True)
            optim_d.step()

        optim_d.zero_grad()
        disc_loss.backward()
        optim_d.step()

        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tGen Loss: {:.6f}, Disc Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), disc_loss.item(), gen_loss.item()))
    # except Exception as e:
        # print(e)