from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import math 
import numpy as np

# def calculate_PSNR(img1, img2, border=0):
#     psnr = peak_signal_noise_ratio(img1, img2, data_range=255.)
#     return psnr

def calculate_PSNR(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

            

def calculate_SSIM(img1, img2):
    score, diff = structural_similarity(img1, img2, full=True, data_range=255.)
    return score, diff
