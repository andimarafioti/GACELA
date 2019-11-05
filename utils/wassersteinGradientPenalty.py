import torch

__author__ = 'Andres'

def calc_gradient_penalty_bayes(discriminator, real_data, fake_data, gamma):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = real_data.size()[0]

    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand(real_data.size()).to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True).to(device)
    disc_interpolates = discriminator(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2) - 1) ** 2) * gamma

    return gradient_penalty