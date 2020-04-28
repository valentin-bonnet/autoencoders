import sys
import os
curr_dir = os.getcwd()
sys.path.append(curr_dir)

import ae
import cvae
import vae
import sbvae
import sbae
import KVAE
import RKN
from KAST import KAST


def get_model(model_type, layers=[32, 64], latent_dim=256, input_shape=64, use_bn=False, std=0.05):

    if model_type == 'AE':
        model = ae.AE(layers, latent_dim, input_shape, use_bn)

    elif model_type == 'CVAE':
        model = cvae.CVAE(layers, latent_dim, input_shape, std, use_bn)

    elif model_type == 'VAE':
        model = vae.VAE(layers, latent_dim)

    elif model_type == 'SBVAE':
        model = sbvae.SBVAE(layers, latent_dim)

    elif model_type == 'SBAE':
        model = sbae.SBAE(layers, latent_dim, input_shape, use_bn)

    elif model_type == 'SBAE_cl':
        model = sbae.SBAE(layers, latent_dim, input_shape, use_bn, classification=True)

    elif model_type == 'KVAE':
        model = KVAE.KVAE(layers, latent_dim, input_shape)

    elif model_type == 'RKN':
        model = RKN.RKN(layers, latent_dim, input_shape)

    elif model_type == 'KAST':
        model = KAST()

    else:
        print("Model type is not good")

    return model