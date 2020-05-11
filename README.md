# GACELA - Generative adversarial context encoder for audio inpainting


We introduce GACELA, a generative adversarial network (GAN) designed to restore missing musical audio data with a duration ranging between hundreds of milliseconds to a few seconds, i.e., to perform long-gap audio inpainting. While previous work either addressed shorter gaps or relied on exemplars by copying available information from other signal parts, GACELA addresses the inpainting of long gaps in two aspects. First, it considers various time scales of audio information by relying on five parallel discriminators with increasing resolution of receptive fields. Second, it is conditioned not only on the available information surrounding the gap, i.e., the context, but also on the latent variable of the conditional GAN. This addresses the inherent multi-modality of audio inpainting at such long gaps and provides the option of user-defined inpainting. GACELA was tested in listening tests on music signals of varying complexity and gap durations ranging from 375ms to 1500ms. While our subjects were often able to detect the inpaintings, the severity of the artifacts decreased from unacceptable to mildly disturbing. GACELA represents a framework capable to integrate future improvements such as processing of more auditory-related features or more explicit musical features.  


# Installation

Install the requirements with `pip install -r requirements.txt`. For windows users, the numpy version should be 1.14.0+mkl (find it [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/)).

The datasets used for the experiments are available:

| Dataset       | Type           | Details  |
| ------------- |:-------------| -----|
| [Lakh](https://colinraffel.com/projects/lmd/) | Midi | Used LMD-matched |
| [Maestro](https://magenta.tensorflow.org/datasets/maestro)      |  Midi & piano | Use full dataset |
| [Free Music Archive](https://github.com/mdeff/fma)|    General music | Used only rock song fom fma-small  |


# Instructions


## Sound examples

- To hear examples please go to the [accompanying website](https://andimarafioti.github.io/GACELA/).


### Acknowledgments

This project accompanies the research work on audio inpainting of large gaps done at the Acoustics Research Institute in Vienna collaborating with the Swiss Data Science Center. The paper was submitted to an IEEE journal as is under review.

We specially thank Michael Mihocic for running the experiments at the Acoustics Research Institute's laboratory during the coronavirus pandemic as well as the subjects for their participation.
