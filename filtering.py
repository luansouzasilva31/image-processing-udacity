import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_filter(image: np.ndarray, custom_kernel: np.ndarray = None,
                 padding: str = 'same'):
    input_image = image.copy()

    kernel = custom_kernel
    if kernel is None:
        print('Set kernel to 5x5 uniform distribution.')
        kernel = np.array([[1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1]])

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
    # TODO: vetorizar
    for j in range(filtered.shape[0]):
        for k in range(filtered.shape[1]):
            region = input_image[j:j+h_kernel, k:k+w_kernel]
            r_conv = np.multiply(kernel, region)

            filtered[j, k] = np.mean(r_conv, axis=(0, 1), keepdims=True)

    return filtered


if __name__ == '__main__':
    path = '/home/luansouzasilva/Imagens/zeze_os_incriveis_capa.jpg'
    image = cv2.imread(path, 0)

    filtered = apply_filter(image)

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(filtered, cmap='gray')
    plt.show(block=True)
