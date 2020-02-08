import models


def get_model(model_type, layers, latent_dim):

    if model_type == 'CVAE':
        model = models.cvae.CVAE(layers=layers, latent_dim=latent_dim)

    elif model_type == 'VAE':
        model = models.vae.VAE(layers, latent_dim)

    elif model_type == 'SBVAE':
        model = models.sbvae.SBVAE(layers, latent_dim)

    elif model_type == 'SBAE':
        model = models.sbae.SBAE(layers, latent_dim)

    else:
        print("Model type is not good")

    return model