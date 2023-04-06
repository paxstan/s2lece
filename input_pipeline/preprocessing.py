import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as tvf
from utils import transform_tools as ttools
import random
from math import ceil


class Scale(object):
    """ Rescale the input PIL.Image to a given size.
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    The smallest dimension of the resulting image will be = size.

    if largest == True: same behaviour for the largest dimension.

    if not can_upscale: don't upscale
    if not can_downscale: don't downscale
    """

    def __init__(self, size, interpolation=Image.BILINEAR, largest=False,
                 can_upscale=True, can_downscale=True):
        assert isinstance(size, int) or (len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.largest = largest
        self.can_upscale = can_upscale
        self.can_downscale = can_downscale

    def __repr__(self):
        fmt_str = "RandomScale(%s" % str(self.size)
        if self.largest: fmt_str += ', largest=True'
        if not self.can_upscale: fmt_str += ', can_upscale=False'
        if not self.can_downscale: fmt_str += ', can_downscale=False'
        return fmt_str + ')'

    def get_params(self, imsize):
        w, h = imsize
        if isinstance(self.size, int):
            cmp = lambda a, b: (a >= b) if self.largest else (a <= b)
            if (cmp(w, h) and w == self.size) or (cmp(h, w) and h == self.size):
                ow, oh = w, h
            elif cmp(w, h):
                ow = self.size
                oh = int(self.size * h / w)
            else:
                oh = self.size
                ow = int(self.size * w / h)
        else:
            ow, oh = self.size
        return ow, oh

    def __call__(self, inp):
        img = ttools.grab_img(inp)
        w, h = img.size

        size2 = ow, oh = self.get_params(img.size)

        if size2 != img.size:
            a1, a2 = img.size, size2
            if (self.can_upscale and min(a1) < min(a2)) or (self.can_downscale and min(a1) > min(a2)):
                img = img.resize(size2, self.interpolation)
                # modifiy range image to adjust for scaling
                img = np.array(img)
                s = min(a1) / min(a2)
                # print(s)
                img = s * img
                img = Image.fromarray(img)

        return ttools.update_img_and_labels(inp, img, persp=(ow / w, 0, 0, 0, oh / h, 0, 0, 0))
        # return img


class RandomScale(Scale):
    """Rescale the input PIL.Image to a random size.
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py
    Args:
        min_size (int): min size of the smaller edge of the picture.
        max_size (int): max size of the smaller edge of the picture.
        ar (float or tuple):
            max change of aspect ratio (width/height).
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, min_size, max_size, ar=1,
                 can_upscale=False, can_downscale=True, interpolation=Image.BILINEAR):
        Scale.__init__(self, 0, can_upscale=can_upscale, can_downscale=can_downscale, interpolation=interpolation)
        assert type(min_size) == type(max_size), 'min_size and max_size can only be 2 ints or 2 floats'
        assert isinstance(min_size, int) and min_size >= 1 or isinstance(min_size, float) and min_size > 0
        assert isinstance(max_size, (int, float)) and min_size <= max_size
        self.min_size = min_size
        self.max_size = max_size
        if type(ar) in (float, int): ar = (min(1 / ar, ar), max(1 / ar, ar))
        assert 0.2 < ar[0] <= ar[1] < 5
        self.ar = ar

    def get_params(self, imsize):
        w, h = imsize
        min_size, max_size = 0, 0
        if isinstance(self.min_size, float):
            min_size = int(self.min_size * min(w, h) + 0.5)
        if isinstance(self.max_size, float):
            max_size = int(self.max_size * min(w, h) + 0.5)
        if isinstance(self.min_size, int):
            min_size = self.min_size
        if isinstance(self.max_size, int):
            max_size = self.max_size

        if not self.can_upscale:
            max_size = min(max_size, min(w, h))

        size = int(0.5 + ttools.rand_log_uniform(min_size, max_size))
        ar = ttools.rand_log_uniform(*self.ar)  # change of aspect ratio

        if w < h:  # image is taller
            ow = size
            oh = int(0.5 + size * h / w / ar)
            if oh < min_size:
                ow, oh = int(0.5 + ow * float(min_size) / oh), min_size
        else:  # image is wider
            oh = size
            ow = int(0.5 + size * w / h * ar)
            if ow < min_size:
                ow, oh = min_size, int(0.5 + oh * float(min_size) / ow)

        assert ow >= min_size, 'image too small (width=%d < min_size=%d)' % (ow, min_size)
        assert oh >= min_size, 'image too small (height=%d < min_size=%d)' % (oh, min_size)
        return ow, oh


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

    def __call__(self, inp):
        img = ttools.grab_img(inp)
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


class StillTransform(object):
    """ Takes and return an image, without changing its shape or geometry.
    """

    def _transform(self, img):
        raise NotImplementedError()

    def __call__(self, inp):
        img = ttools.grab_img(inp)

        # transform the image (size should not change)
        try:
            img = self._transform(img)
        except TypeError:
            pass

        return ttools.update_img_and_labels(inp, img, persp=(1, 0, 0, 0, 1, 0, 0, 0))


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
        return Image.fromarray(np.uint8(img.clip(0, 255)))


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

        # if saturation > 0:
        #     saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
        #     transforms.append(tvf.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        # if hue > 0:
        #     hue_factor = np.random.uniform(-hue, hue)
        #     transforms.append(tvf.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

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


class RandomTranslation(object):
    def __init__(self, roll):
        self.roll = np.random.uniform(-roll, roll)

    def __call__(self, inp):
        img = ttools.grab_img(inp)
        trf = ttools.translate(self.roll, 0)
        img = img.transform(img.size, Image.PERSPECTIVE, (1, 0, -self.roll, 0, 1, 0, 0, 0))
        return ttools.update_img_and_labels(inp, img, persp=trf)


class RandomCrop(object):
    """Crop the given PIL Image at a random location.
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __repr__(self):
        return "RandomCrop(%s)" % str(self.size)

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        assert h >= th and w >= tw, "Image of %dx%d is too small for crop %dx%d" % (w, h, tw, th)

        y = np.random.randint(0, h - th) if h > th else 0
        x = np.random.randint(0, w - tw) if w > tw else 0
        return x, y, tw, th

    def __call__(self, inp):
        img = ttools.grab_img(inp)

        padl = padt = 0
        if self.padding:
            if ttools.is_pil_image(img):
                img = ImageOps.expand(img, border=self.padding, fill=0)
            else:
                assert isinstance(img, ttools.DummyImg)
                img = img.expand(border=self.padding)
            if isinstance(self.padding, int):
                padl = padt = self.padding
            else:
                padl, padt = self.padding[0:2]

        i, j, tw, th = self.get_params(img, self.size)
        img = img.crop((i, j, i + tw, j + th))

        return ttools.update_img_and_labels(inp, img, persp=(1, 0, padl - i, 0, 1, padt - j, 0, 0))
