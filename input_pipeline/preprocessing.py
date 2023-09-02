import numpy as np
import cv2
import torchvision.transforms as tvf

# img_mean = [5.4289184]
# img_std = [9.20105]
img_mean = [0.5]
img_std = [0.5]

normalize_img = tvf.Compose([tvf.ToTensor()])


def preprocess_range_image_old(img_array):
    # normalized_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    # normalized_array = 1 - normalized_arra
    # normalized_array = (normalized_array * 255).astype(np.uint8)

    # Define the neighborhood size for computing the local average
    neighborhood_size = (8, 8)

    # Compute the local average color using a box filter
    local_average = cv2.boxFilter(img_array, -1, neighborhood_size)

    # Subtract the local average from the original image
    image_local_normal = img_array - local_average

    local_mean = np.mean(image_local_normal)

    # Normalize the subtracted image to the range [0, 255]
    # image_normalized = (image_subtracted - np.min(image_subtracted)) / (
    #         np.max(image_subtracted) - np.min(image_subtracted))
    # image_normalized = (image_normalized * 255).astype(np.uint8)

    # Adjust the image so that the local average is mapped to 50% gray
    image_local_normal = (image_local_normal + 0.5 * local_mean)
    image_normalized = (image_local_normal - np.min(image_local_normal)) / (
            np.max(image_local_normal) - np.min(image_local_normal))
    image_normalized = (image_normalized * 255).astype(np.uint8)

    denoised_image = cv2.fastNlMeansDenoising(
        image_normalized, None, h=5, templateWindowSize=7, searchWindowSize=16)
    edges = cv2.Canny(denoised_image, threshold1=100, threshold2=150)

    sobel_x = cv2.Sobel(image_normalized, cv2.CV_64F, 1, 0, ksize=3)

    # Apply Sobel operator to find vertical edges
    sobel_y = cv2.Sobel(image_normalized, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # amplified_edges = cv2.addWeighted(denoised_image, 0.7, edges, 0.3, 0)

    # fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)
    # axs[0].imshow(img_array, cmap='gray')
    # axs[0].axis('off')
    # axs[1].imshow(denoised_image, cmap='gray')
    # axs[1].axis('off')
    # axs[2].imshow(amplified_edges, cmap='gray')
    # axs[2].axis('off')
    # axs[3].imshow(edges, cmap='gray')
    # axs[3].axis('off')
    #
    # plt.show()

    # image_normalized = 2.0 * (image_normalized - 0.5)

    normalized_gradient = (gradient_magnitude - np.min(gradient_magnitude)) / (
            np.max(gradient_magnitude) - np.min(gradient_magnitude))
    normalized_gradient = normalize_img(normalized_gradient)

    return normalize_img(img_array), normalized_gradient


def preprocess_range_image(img_array):
    image_normalized = (img_array - np.min(img_array)) / (
            np.max(img_array) - np.min(img_array))
    image_normalized = (image_normalized * 255).astype(np.uint8)

    sobel_x = cv2.Sobel(image_normalized, cv2.CV_64F, 1, 0, ksize=3)

    # Apply Sobel operator to find vertical edges
    sobel_y = cv2.Sobel(image_normalized, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    normalized_gradient = (gradient_magnitude - np.min(gradient_magnitude)) / (
            np.max(gradient_magnitude) - np.min(gradient_magnitude))
    normalized_gradient = normalize_img(normalized_gradient)

    return img_array, normalized_gradient
