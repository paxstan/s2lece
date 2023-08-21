import numpy as np
import cv2
import torch
import torchvision.transforms as tvf
from utils import transform_tools as ttools
from PIL import Image
import random
from math import ceil

# img_mean = [5.4289184]
# img_std = [9.20105]
img_mean = [0.5]
img_std = [0.5]


normalize_img = tvf.Compose([tvf.ToTensor()])


# normalize_img = tvf.Compose([tvf.ToTensor()])

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


class StillTransform(object):
    """ Takes and return an image, without changing its shape or geometry.
    """

    def _transform(self, img):
        raise NotImplementedError()

    def __call__(self, img):
        img = self._transform(img)
        return img


class PixelNoise(StillTransform):
    """ Takes an image, and add random white noise.
    """

    def __init__(self, ampl=20):
        StillTransform.__init__(self)
        assert 0 <= ampl < 255
        self.ampl = ampl

    def __repr__(self):
        return "PixelNoise(%g)" % self.ampl

    def _transform(self, img):
        img = np.float32(img)
        img = img + np.random.uniform(0.5 - self.ampl / 2, 0.5 + self.ampl / 2, size=img.shape)
        return np.uint8(img.clip(0, 255))


class ColorJitter(StillTransform):
    """Randomly change the brightness, contrast and saturation of an image.
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __repr__(self):
        return "ColorJitter(%g,%g,%g,%g)" % (
            self.brightness, self.contrast, self.saturation, self.hue)

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
        Transform which randomly adjusts brightness, contrast and
        saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(tvf.Lambda(lambda img: ttools.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(tvf.Lambda(lambda img: ttools.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(tvf.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(tvf.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        # print('colorjitter: ', brightness_factor, contrast_factor, saturation_factor, hue_factor) # to debug random seed

        np.random.shuffle(transforms)
        transform = tvf.Compose(transforms)
        return transform

    def _transform(self, img):
        transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        # r, i, re = img.split()
        # i_tf = transform(i)
        # re_tf = transform(re)
        # img = Image.merge("RGB", (r, i_tf, re_tf))
        return transform(img)


class RandomTilting(object):
    """Apply a random tilting (left, right, up, down) to the input PIL.Image
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    Args:
        maginitude (float):
            maximum magnitude of the random skew (value between 0 and 1)
        directions (string):
            tilting directions allowed (all, left, right, up, down)
            examples: "all", "left,right", "up-down-right"
    """

    def __init__(self, magnitude, directions='all'):
        self.magnitude = magnitude
        self.directions = directions.lower().replace(',', ' ').replace('-', ' ')

    def __repr__(self):
        return "RandomTilt(%g, '%s')" % (self.magnitude, self.directions)

    def __call__(self, img):
        img = Image.fromarray(img)
        w, h = img.size

        x1, y1, x2, y2 = 0, 0, h, w
        new_plane = original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]

        max_skew_amount = max(w, h)
        max_skew_amount = int(ceil(max_skew_amount * self.magnitude))
        skew_amount = random.randint(1, max_skew_amount)

        if self.directions == 'all':
            choices = [0, 1, 2, 3]
        else:
            dirs = ['left', 'right', 'up', 'down']
            choices = []
            for d in self.directions.split():
                try:
                    choices.append(dirs.index(d))
                except:
                    raise ValueError('Tilting direction %s not recognized' % d)

        skew_direction = random.choice(choices)

        # print('randomtitlting: ', skew_amount, skew_direction) # to debug random

        if skew_direction == 0:
            # Left Tilt
            new_plane = [(y1, x1 - skew_amount),  # Top Left
                         (y2, x1),  # Top Right
                         (y2, x2),  # Bottom Right
                         (y1, x2 + skew_amount)]  # Bottom Left
        elif skew_direction == 1:
            # Right Tilt
            new_plane = [(y1, x1),  # Top Left
                         (y2, x1 - skew_amount),  # Top Right
                         (y2, x2 + skew_amount),  # Bottom Right
                         (y1, x2)]  # Bottom Left
        elif skew_direction == 2:
            # Forward Tilt
            new_plane = [(y1 - skew_amount, x1),  # Top Left
                         (y2 + skew_amount, x1),  # Top Right
                         (y2, x2),  # Bottom Right
                         (y1, x2)]  # Bottom Left
        elif skew_direction == 3:
            # Backward Tilt
            new_plane = [(y1, x1),  # Top Left
                         (y2, x1),  # Top Right
                         (y2 + skew_amount, x2),  # Bottom Right
                         (y1 - skew_amount, x2)]  # Bottom Left

        # To calculate the coefficients required by PIL for the perspective skew,
        # see the following Stack Overflow discussion: https://goo.gl/sSgJdj
        matrix = []

        for p1, p2 in zip(new_plane, original_plane):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        A = np.matrix(matrix, dtype=float)
        B = np.array(original_plane).reshape(8)

        homography = np.dot(np.linalg.pinv(A), B)
        homography = tuple(np.array(homography).reshape(8))
        print(homography)

        img = img.transform(img.size, Image.PERSPECTIVE, homography, resample=Image.BICUBIC)

        homography = np.linalg.pinv(np.float32(homography + (1,)).reshape(3, 3)).ravel()[:8]
        return ttools.update_img_and_labels(inp, img, persp=tuple(homography))
