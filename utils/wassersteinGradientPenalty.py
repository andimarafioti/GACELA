import torch

__author__ = 'Andres'


def calc_gradient_penalty(netD, real_data, fake_data, gamma):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = real_data.size()[0]
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data.detach() + (1 - alpha) * fake_data.detach()

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = gamma * ((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty

# import tensorflow as tf

def wgan_regularization(discriminator, real, fake, gamma):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # calculate `x_hat`
    batch_size = real.size()[0]
    # eps = tf.random.uniform(shape=[batch_size], minval=0, maxval=1)
    eps = torch.rand(batch_size, 1, 1, 1)

    eps = eps.expand(real.size()).to(device)
    x_hat = eps * real + (1.0 - eps) * fake
    x_hat.requires_grad_(True)

    D_x_hat = discriminator(x_hat)

    gradients = torch.autograd.grad(outputs=D_x_hat, inputs=x_hat,
                                    grad_outputs=torch.ones(D_x_hat.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    # print(gradients)
    gradients = gradients.view(gradients.size(0), -1)
    # print(gradients.norm(2, dim=1).mean())
    gradient_penalty = gamma * ((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty


import torch.autograd as autograd


def calc_gradient_penalty_bayes(discriminator, real_data, fake_data, gamma):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = real_data.size()[0]

    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand(real_data.size()).to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = autograd.Variable(interpolates, requires_grad=True).to(device)

    disc_interpolates = discriminator(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gamma

    return gradient_penalty