import cv2
import numpy as np

# Grayscale
def BGR2GRAY(img):
    # Grayscale
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray


def floyd_steinberg(source, q):
    i = 255
    for y in range(1, source.shape[0] - 1):
        for x in range(1, source.shape[1] - 1):
            oldpixel = source[y][x]
            newpixel = np.round(source[y][x] / (i / q)) * (i / q)
            source[y][x] = newpixel

            quant_error = oldpixel - newpixel

            source[y, x + 1] = source[y, x + 1] + quant_error * 7 / 16
            source[y + 1, x] = source[y + 1, x] + quant_error * 5 / 16
            source[y + 1, x - 1] = source[y + 1, x - 1] + quant_error * 3 / 16
            source[y + 1, x + 1] = source[y + 1, x + 1] + quant_error * 1 / 16

    return source


def quantization(source, q):
    i = 255
    quantized = np.round(source / (i / q)) * (i / q)

    return quantized


if __name__ == '__main__':
    path = './images/dithering/1.png'
    q = 4

    source = cv2.imread(path)
    source = BGR2GRAY(source)

    quantized = quantization(source, q)
    cv2.imwrite(f'quantized_{q}.png', quantized)

    dithered = floyd_steinberg(source, q)
    cv2.imwrite(f'dithered_{q}.png', dithered)
