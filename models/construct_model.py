import sys
import os
curr_dir = os.getcwd()
sys.path.append(curr_dir)

import ae
import cvae
import vae
import sbvae
import sbae


def get_model(model_type, layers, latent_dim, input_shape):

    if model_type == 'AE':
        model = ae.AE(layers, latent_dim, input_shape)

    elif model_type == 'CVAE':
        model = cvae.CVAE(layers=layers, latent_dim=latent_dim)

    elif model_type == 'VAE':
        model = vae.VAE(layers, latent_dim)

    elif model_type == 'SBVAE':
        model = sbvae.SBVAE(layers, latent_dim)

    elif model_type == 'SBAE':
        model = sbae.SBAE(layers, latent_dim, input_shape)

    else:
        print("Model type is not good")

    return model