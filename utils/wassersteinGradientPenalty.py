import torch

__author__ = 'Andres'


def calc_gradient_penalty(netD, real_data, fake_data, gamma):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = real_data.size()[0]
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to('cuda')
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = gamma * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
