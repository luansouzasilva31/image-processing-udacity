import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_kernel(ksize: int, sigma: float):
    ax = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(xx, yy, kernel, rstride=1, cstride=1,
                           antialiased=True, facecolor='red')
    plt.show()

    return kernel / np.sum(kernel)


def uniform_kernel(ksize: int):
    kernel = np.ones((ksize, ksize))

    return kernel


def apply_filter(image: np.ndarray, custom_kernel: np.ndarray = None,
                 padding: str = 'same'):
    # TODO: include stride
    input_image = image.copy()

    kernel = custom_kernel
    if kernel is None:
        print('Set kernel to 5x5 uniform distribution.')
        kernel = uniform_kernel((31, 31))

    h_kernel, w_kernel = kernel.shape[:2]
    ij, ik = h_kernel // 2, w_kernel // 2  # indexes

    if padding == 'valid':  # shape reduction
        shape = input_image.shape[:2] - np.array([ij, ik])
    elif padding == 'same':
        shape = input_image.shape[:2]
        input_image = np.pad(input_image, [(ij, ij), (ik, ik)],
                             mode='constant')
    else:
        raise NotImplementedError(
            f'padding must be `valid` or `same`, not {padding}')

    filtered = np.zeros(shape, dtype=np.uint8)

    # apply kernel
    # # TODO: vetorizar
    # for j in range(filtered.shape[0]):
    #     for k in range(filtered.shape[1]):
    #         region = input_image[j:j+h_kernel, k:k+w_kernel]
    #         r_conv = np.multiply(kernel, region)

    #         filtered[j, k] = np.mean(r_conv, axis=(0, 1), keepdims=True)

    # get regions
    regions = np.zeros((filtered.shape[0], filtered.shape[1], 
                        h_kernel, w_kernel))
    for i in range(filtered.shape[0]):
        for j in range(filtered.shape[1]):
            regions[i, j] = input_image[i:i+h_kernel, j:j+w_kernel]
    i_kernel = np.expand_dims(kernel, axis=(0, 1))

    # apply vectorization
    filtered = np.multiply(i_kernel, regions)
    filtered = np.mean(filtered, axis=(-2, -1)).astype(np.uint8)

    return filtered


if __name__ == '__main__':
    path = 'c:\\Users\\luans\\Pictures\\pombo.jpg'
    image = cv2.imread(path, 0)

    filtered = apply_filter(image)

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(filtered, cmap='gray')
    plt.show(block=True)
