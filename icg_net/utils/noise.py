import numpy as np
import skimage.transform


def apply_dex_noise(img, gamma_shape=1000, gamma_scale=0.001, gp_sigma=0.005, gp_scale=4.0):
    gamma_noise = np.random.gamma(gamma_shape, gamma_scale)
    img = img * gamma_noise

    h, w = img.shape[:2]
    gp_sample_height = int(h / gp_scale)
    gp_sample_width = int(w / gp_scale)
    gp_noise = np.random.randn(gp_sample_height, gp_sample_width) * gp_sigma
    gp_noise = skimage.transform.resize(gp_noise, img.shape[:2], order=1, anti_aliasing=False, mode="constant")

    img += gp_noise
    return img
