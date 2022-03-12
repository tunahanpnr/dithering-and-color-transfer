import math
import cv2
import numpy as np


def image_filter(src, mat):
    res = np.zeros(src.shape)
    for x in range(res.shape[0]):
        for y in range(res.shape[1]):
            res[x][y] = np.dot(mat, src[x][y])

    return res


def log10(src):
    res = src.copy()
    for x in range(res.shape[0]):
        for y in range(res.shape[1]):
            res[x][y] = [1 if p == 0 else p for p in res[x][y]]

    return np.log10(res)


def mean_std(src):
    means, stds = cv2.mean(src)[0:3], [np.std(src[:, :, 0]), np.std(src[:, :, 1]), np.std(src[:, :, 2])]
    return np.asarray(means), np.asarray(stds)


def colorTransfer(source, target):
    rgb_to_lms_filter = np.array([[0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]])

    b = np.array([[1 / math.sqrt(3), 0, 0], [0, 1 / math.sqrt(6), 0], [0, 0, 1 / math.sqrt(2)]])
    c = np.array([[1, 1, 1], [1, 1, - 2], [1, - 1, 0]])
    lms_to_lab_filter = np.dot(b, c)

    b2 = np.array([[math.sqrt(3) / 3, 0, 0], [0, math.sqrt(6) / 6, 0], [0, 0, math.sqrt(2) / 2]])
    c2 = np.array([[1, 1, 1], [1, 1, - 1], [1, -2, 0]])
    lab_to_lms_filter = np.dot(c2, b2)

    lms_to_rgb_filter = np.array([[4.4679, -3.5873, 0.1193], [-1.2186, 2.3809, -0.1624], [0.0497, -0.2439, 1.2045]])

    # convert source to LMS
    LMS_s = image_filter(source, rgb_to_lms_filter)
    LMS_t = image_filter(target, rgb_to_lms_filter)

    # log LMS values
    LMS_s = log10(LMS_s)
    LMS_t = log10(LMS_t)

    # convert to lab space
    LAB_s = image_filter(LMS_s, lms_to_lab_filter)
    LAB_t = image_filter(LMS_t, lms_to_lab_filter)

    # mean and std
    s_means, s_stds = mean_std(LAB_s)
    t_means, t_stds = mean_std(LAB_t)

    # Subtract the mean of source image from the source image
    l_sub_mean = LAB_s[:, :, 0] - s_means[0]
    a_sub_mean = LAB_s[:, :, 1] - s_means[1]
    b_sub_mean = LAB_s[:, :, 2] - s_means[2]

    # Scale the data points and add target's mean
    l_scale = ((t_stds[0] / s_stds[0]) * l_sub_mean) + t_means[0]
    a_scale = ((t_stds[1] / s_stds[1]) * a_sub_mean) + t_means[1]
    b_scale = ((t_stds[2] / s_stds[2]) * b_sub_mean) + t_means[2]

    # Apply transform matrix to convert lαβ to LMS
    lab = image_filter(cv2.merge([l_scale, a_scale, b_scale]), lab_to_lms_filter)

    #  Go back to linear space
    lab = np.power(10, lab)

    #  Apply transform matrix to convert LMS to RGB
    rgb = image_filter(lab, lms_to_rgb_filter)

    # rgb2bgr
    bgr = cv2.merge([rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0]])

    return bgr


if __name__ == '__main__':
    path_source = './images/colortransfer/scotland_house.jpg'
    path_target = './images/colortransfer/scotland_plain.jpg'

    # read image and convert to rgb
    source = cv2.imread(path_source)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    target = cv2.imread(path_target)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    # perform color transferring
    new_rgb_img = colorTransfer(source, target)
    cv2.imwrite('output.jpg', new_rgb_img)