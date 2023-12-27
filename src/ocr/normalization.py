import cv2
import numpy as np

from src.ocr.helpers import resize


class HysterThresh:
    def __init__(self, img):
        img = 255 - img
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        hist, bins = np.histogram(img.ravel(), 256, [0, 256])

        self.high = np.argmax(hist) + 65
        self.low = np.argmax(hist) + 45
        self.diff = 255 - self.high

        self.img = img
        self.im = np.zeros(img.shape, dtype=img.dtype)

    def get_image(self):
        self._hyster()
        return np.uint8(self.im)

    def _hyster_rec(self, r, c):
        h, w = self.img.shape
        for ri in range(r - 1, r + 2):
            for ci in range(c - 1, c + 2):
                if (
                    h > ri >= 0
                    and w > ci >= 0
                    and self.im[ri, ci] == 0
                    and self.high > self.img[ri, ci] >= self.low
                ):
                    self.im[ri, ci] = self.img[ri, ci] + self.diff
                    self._hyster_rec(ri, ci)

    def _hyster(self):
        r, c = self.img.shape
        for ri in range(r):
            for ci in range(c):
                if self.img[ri, ci] >= self.high:
                    self.im[ri, ci] = 255
                    self.img[ri, ci] = 255
                    self._hyster_rec(ri, ci)


def word_normalization(
    image, height, border=True, tilt=True, border_size=15, hyst_norm=False
):
    """Preprocess a word - resize, binarize, tilt world."""
    image = resize(image, height, True)

    if hyst_norm:
        th = _hyst_word_norm(image)
    else:
        img = cv2.bilateralFilter(image, 10, 30, 30)
        gray = 255 - cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        ret, th = cv2.threshold(norm, 50, 255, cv2.THRESH_TOZERO)

    if tilt:
        return _word_tilt(th, height, border, border_size)
    return _crop_add_border(th, height=height, border=border, border_size=border_size)


def _hyst_word_norm(image):
    """Word normalization using hystheresis thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     img = cv2.bilateralFilter(gray, 0, 10, 30)
    img = cv2.bilateralFilter(gray, 10, 10, 30)
    return HysterThresh(img).get_image()


def _word_tilt(img, height, border=True, border_size=15):
    """Detect the angle and tilt the image."""
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 30)

    if lines is not None:
        mean_angle = 0
        # Set min number of valid lines (try higher)
        num_lines = np.sum(1 for l in lines if l[0][1] < 0.7 or l[0][1] > 2.6)
        if num_lines > 1:
            mean_angle = np.mean(
                [l[0][1] for l in lines if l[0][1] < 0.7 or l[0][1] > 2.6]
            )

        # Look for angle with correct value
        if mean_angle != 0 and (mean_angle < 0.7 or mean_angle > 2.6):
            img = _tilt_by_angle(img, mean_angle, height)
    return _crop_add_border(img, height, 50, border, border_size)


def _tilt_by_angle(img, angle, height):
    """Tilt the image by given angle."""
    dist = np.tan(angle) * height
    width = len(img[0])
    s_points = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

    # Dist is positive for angle < 0.7; negative for angle > 2.6
    # Image must be shifed to right
    if dist > 0:
        t_points = np.float32(
            [[0, 0], [dist, height], [width + dist, height], [width, 0]]
        )
    else:
        t_points = np.float32(
            [[-dist, 0], [0, height], [width, height], [width - dist, 0]]
        )

    M = cv2.getPerspectiveTransform(s_points, t_points)
    return cv2.warpPerspective(img, M, (int(width + abs(dist)), height))


def _crop_add_border(img, height, threshold=50, border=True, border_size=15):
    """Crop and add border to word image of letter segmentation."""
    # Clear small values

    ret, img = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)

    x0 = 0
    y0 = 0
    x1 = img.shape[1]
    y1 = img.shape[0]

    for i in range(img.shape[0]):
        if np.count_nonzero(img[i, :]) > 1:
            y0 = i
            break
    for i in reversed(range(img.shape[0])):
        if np.count_nonzero(img[i, :]) > 1:
            y1 = i + 1
            break
    for i in range(img.shape[1]):
        if np.count_nonzero(img[:, i]) > 1:
            x0 = i
            break
    for i in reversed(range(img.shape[1])):
        if np.count_nonzero(img[:, i]) > 1:
            x1 = i + 1
            break

    if height != 0:
        img = resize(img[y0:y1, x0:x1], height, True)
    else:
        img = img[y0:y1, x0:x1]

    if border:
        return cv2.copyMakeBorder(
            img, 0, 0, border_size, border_size, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    return img
