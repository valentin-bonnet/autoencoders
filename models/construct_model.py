import sys

sys.path.append('/home/valentin/Programmation/Projects/vae/models')

import cvae
import vae
import sbvae
import sbae


def get_model(model_type, layers, latent_dim):

    if model_type == 'CVAE':
        model = cvae.CVAE(layers=layers, latent_dim=latent_dim)

    elif model_type == 'VAE':
        model = vae.VAE(layers, latent_dim)

    elif model_type == 'SBVAE':
        model = sbvae.SBVAE(layers, latent_dim)

    elif model_type == 'SBAE':
        model = sbae.SBAE(layers, latent_dim)

    else:
        print("Model type is not good")

    return model