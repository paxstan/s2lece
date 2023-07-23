import numpy as np
import cv2
import torchvision.transforms as tvf
import matplotlib.pyplot as plt
from utils import transform_tools as ttools
import random
from math import ceil

# img_mean = [5.4289184]
# img_std = [9.20105]
# img_mean = [0.02]
# img_std = [0.05]

# normalize_img = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=img_mean, std=img_std)])
normalize_img = tvf.Compose([tvf.ToTensor()])


# normalize_img = tvf.Compose([tvf.ToTensor()])

def preprocess_range_image(img_array):
    img_array[img_array < 0] = 0
    # normalized_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    # normalized_array = 1 - normalized_arra
    # normalized_array = (normalized_array * 255).astype(np.uint8)

    # Define the neighborhood size for computing the local average
    neighborhood_size = (15, 15)

    # Compute the local average color using a box filter
    local_average = cv2.boxFilter(img_array, -1, neighborhood_size)

    # Subtract the local average from the original image
    image_subtracted = img_array - local_average

    # Normalize the subtracted image to the range [0, 255]
    image_normalized = (image_subtracted - np.min(image_subtracted)) / (
            np.max(image_subtracted) - np.min(image_subtracted))
    image_normalized = (image_normalized * 255).astype(np.uint8)

    # Adjust the image so that the local average is mapped to 50% gray
    image_adjusted = (image_normalized + 0.5 * (128 - np.mean(image_normalized))).astype(np.uint8)

    # denoise_bi = cv2.bilateralFilter(image_adjusted, d=5, sigmaColor=50, sigmaSpace=32)
    denoised_image = cv2.fastNlMeansDenoising(image_adjusted, None, h=10, templateWindowSize=7, searchWindowSize=16)
    # edges_noise = cv2.Canny(normalized_array, threshold1=20, threshold2=150)
    edges = cv2.Canny(denoised_image, threshold1=20, threshold2=150)
    edge_map = edges * 255
    amplified_edges = cv2.addWeighted(denoised_image, 1, edge_map, 1, 0)

    fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)
    axs[0].imshow(img_array, cmap='gray')
    axs[0].axis('off')
    axs[1].imshow(image_adjusted, cmap='gray')
    axs[1].axis('off')
    axs[2].imshow(denoised_image, cmap='gray')
    axs[2].axis('off')
    axs[3].imshow(edges, cmap='gray')
    axs[3].axis('off')

    plt.show()

    img = normalize_img(denoised_image)

    return img
