import torch
from torch import nn

from model.borderEncoder import BorderEncoder
from model.discriminator import Discriminator
from model.generator import Generator

import time

from utils.colorize import colorize
from utils.consistencyComputer import consistency
from utils.torchModelSaver import TorchModelSaver
from utils.wassersteinGradientPenalty import calc_gradient_penalty_bayes

__author__ = 'Andres'


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


def train(args, device, train_loader, epoch, summary_writer, batch_idx=0):
    discriminators = nn.ModuleList(
        [Discriminator(args['discriminator'], args['discriminator_in_shape']) for _ in range(3)]
    ).to(device)

    left_border_encoder = BorderEncoder(args['borderEncoder']).to(device)
    right_border_encoder = BorderEncoder(args['borderEncoder']).to(device)

    generator = Generator(args['generator'], args['generator_input']).to(device)

    optim_g = torch.optim.Adam(list(generator.parameters()) + list(left_border_encoder.parameters()) +
                               list(right_border_encoder.parameters()),
                               lr=args['optimizer']['generator']['learning_rate'],
                               betas=(0.5, 0.9))
    optims_d = [torch.optim.Adam(discriminator.parameters(),
                                 lr=args['optimizer']['discriminator']['learning_rate'],
                                 betas=(0.5, 0.9)) for discriminator in discriminators]

    model_saver = TorchModelSaver(args['experiment_name'], args['save_path'])

    if batch_idx == 0 and epoch == 0:
        model_saver.makeFolder()
        discriminators.apply(init_weights)
        left_border_encoder.apply(init_weights)
        right_border_encoder.apply(init_weights)
        generator.apply(init_weights)
    else:
        generator, discriminators, left_border_encoder, right_border_encoder, optim_g, optims_d = \
            model_saver.loadModel(generator, discriminators, left_border_encoder, right_border_encoder, optim_g,
                                  optims_d, batch_idx, epoch)

    print('try')
    start_time = time.time()
    prev_iter_time = start_time
    d_loss_gp_to_summarize = 0

    try:
        for batch_idx, data in enumerate(train_loader, batch_idx):
            data = data.to(device).float()
            data = data.view(args['optimizer']['batch_size'], *args['spectrogram_shape'])
            real_spectrograms = data[::2]
            fake_left_borders = data[1::2, :, :, :args['split'][0]]
            fake_right_borders = data[1::2, :, :, args['split'][0] + args['split'][1]:]

            # optimize D
            for _ in range(args['optimizer']['n_critic']):
                for index, (discriminator, optim_d) in enumerate(zip(discriminators, optims_d)):
                    optim_d.zero_grad()
                    encoded_left_border = left_border_encoder(fake_left_borders)
                    encoded_right_border = right_border_encoder(fake_right_borders)
                    encoded_size = encoded_left_border.size()
                    noise = torch.rand(encoded_size[0], 4, encoded_size[2], encoded_size[3]).to(device)
                    generated_spectrograms = generator(torch.cat((encoded_left_border, encoded_right_border, noise), 1))

                    fake_spectrograms = torch.cat((fake_left_borders, generated_spectrograms, fake_right_borders), 3)
                    scale = 2 ** index
                    time_axis = args['spectrogram_shape'][2]
                    start = int((time_axis - (time_axis // 4) * scale) / 2)
                    end = time_axis - start
                    x_fake = fake_spectrograms[:, :, :, start:end:scale].detach()
                    x_real = real_spectrograms[:, :, :, start:end:scale].detach()

                    d_loss_r = torch.mean(discriminator(x_real))
                    d_loss_f = torch.mean(discriminator(x_fake))

                    grad_pen = calc_gradient_penalty_bayes(discriminator, x_real, x_fake, args['gamma_gp'])
                    d_loss_gp = torch.mean(grad_pen)
                    d_loss_gp_to_summarize += d_loss_gp.data
                    disc_loss = d_loss_f - d_loss_r + d_loss_gp

                    disc_loss.backward()
                    optim_d.step()

            # optimize G

            optim_g.zero_grad()

            encoded_left_border = left_border_encoder(fake_left_borders)
            encoded_right_border = right_border_encoder(fake_right_borders)
            encoded_size = encoded_left_border.size()
            noise = torch.rand(encoded_size[0], 4, encoded_size[2], encoded_size[3]).to(device)
            generated_spectrograms = generator(torch.cat((encoded_left_border, encoded_right_border, noise), 1))

            fake_spectrograms = torch.cat((fake_left_borders, generated_spectrograms, fake_right_borders), 3)
            d_loss_f = 0

            for index, discriminator in enumerate(discriminators):
                scale = 2 ** index
                time_axis = args['spectrogram_shape'][2]
                start = int((time_axis - (time_axis // 4) * scale) / 2)
                end = time_axis - start
                x_fake = fake_spectrograms[:, :, :, start:end:scale]
                d_loss_f += torch.mean(discriminator(x_fake))

            gen_loss = - d_loss_f
            gen_loss.backward()
            optim_g.step()

            if batch_idx % args['log_interval'] == 0:
                current_time = time.time()

                print(" * Epoch: [{:2d}] [{:4d}/{:4d} ({:.0f}%)] "
                      "Counter:{:2d}\t"
                      "({:4.1f} min\t"
                      "{:4.3f} examples/sec\t"
                      "{:4.2f} sec/batch)\n"
                      "   Disc batch loss:{:.8f}\t"
                      "   Gen batch loss:{:.8f}\t"
                      "   Reg batch :{:.8f}\t".format(
                    int(epoch),
                    int(batch_idx * len(data)),
                    int(len(train_loader.dataset) / len(data)), 100. * batch_idx / len(train_loader), int(batch_idx),
                                                                (current_time - start_time) / 60,
                                                                args['log_interval'] * args['optimizer']['batch_size'] / (
                                                                current_time - prev_iter_time),
                                                                (current_time - prev_iter_time) / args['log_interval'],
                    disc_loss.item(),
                    gen_loss.item(),
                    d_loss_gp.item()))
                prev_iter_time = current_time
            if batch_idx % args['tensorboard_interval'] == 0:
                summary_writer.add_scalar("Disc/Neg_Loss", -disc_loss, global_step=batch_idx)
                summary_writer.add_scalar("Disc/Neg_Critic", d_loss_f - d_loss_r, global_step=batch_idx)
                summary_writer.add_scalar("Disc/Loss_f", d_loss_f, global_step=batch_idx)
                summary_writer.add_scalar("Disc/Loss_r", d_loss_r, global_step=batch_idx)
                summary_writer.add_scalar("Gen/Loss", gen_loss, global_step=batch_idx)
                real_c = consistency((real_spectrograms - 1) * 5)
                fake_c = consistency((generated_spectrograms - 1) * 5)

                mean_R_Con, std_R_Con = real_c.mean(), real_c.std()
                mean_F_Con, std_F_Con = fake_c.mean(), fake_c.std()

                summary_writer.add_scalar("Gen/Reg", torch.abs(mean_R_Con - mean_F_Con), global_step=batch_idx)
                summary_writer.add_scalar("Gen/F_Con", mean_F_Con, global_step=batch_idx)
                summary_writer.add_scalar("Gen/F_STD_Con", std_F_Con, global_step=batch_idx)
                summary_writer.add_scalar("Gen/R_Con", mean_R_Con, global_step=batch_idx)
                summary_writer.add_scalar("Gen/R_STD_Con", std_R_Con, global_step=batch_idx)
                summary_writer.add_scalar("Gen/STD_diff", torch.abs(std_F_Con - std_R_Con), global_step=batch_idx)

                summary_writer.add_scalar("Disc/GradPen",
                                          d_loss_gp_to_summarize / args['tensorboard_interval'] / args['optimizer'][
                                              'n_critic'], global_step=batch_idx)
                d_loss_gp_to_summarize = 0

                for index in range(4):
                    summary_writer.add_image("images/Real_Image/" + str(index), colorize(real_spectrograms[index]),
                                             global_step=batch_idx)
                    summary_writer.add_image("images/Fake_Image/" + str(index), colorize(fake_spectrograms[index], -1, 1),
                                             global_step=batch_idx)
            if batch_idx % args['save_interval'] == 0:
                model_saver.saveModel(generator, discriminators, left_border_encoder, right_border_encoder, optim_g,
                                      optims_d, batch_idx, epoch)
        can_restart = True
        return batch_idx, can_restart
    except KeyboardInterrupt:
        model_saver.saveModel(generator, discriminators, left_border_encoder, right_border_encoder, optim_g,
                              optims_d, batch_idx, epoch)
        should_restart = False
        return batch_idx, should_restart
