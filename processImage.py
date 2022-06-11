import cv2, numpy as np, sys


def removeDuplicates(image):
    """
        Remove duplicate rows in image (numpy array)
        return new image
    """
    l = [image[0]]
    for i in range(1, image.shape[0]):
        if not np.array_equal(image[i], image[i - 1]):  # Prev != Current
            l.append(image[i])
    return np.asarray(l, dtype=image.dtype)


def transpose(image):
    """
        Transpose the given image
    """
    x, y, z = image.shape
    transpose_shape = (y, x, z)
    transpose = np.empty(shape=transpose_shape)
    for i in range(x):
        for j in range(y):
            transpose[j, i] = image[i, j]
    return transpose


def preProcess(image: np.ndarray):
    """
        Remove duplicate rows and columns 
    """
    image = removeDuplicates(image)
    image = transpose(image)
    image = removeDuplicates(image)
    image = transpose(image)
    return image


def postProcess(image, size):
    """
        Scale Image to (size) by usinge Nearest Neighbour algorithm
    """
    image = cv2.resize(image, dsize=size, interpolation=cv2.INTER_NEAREST)
    return image


if __name__ == "__main__":
    FILE = sys.argv[1]
    img = cv2.imread(FILE)
    size = img.shape[:2]
    img = preProcess(img)
    img = postProcess(img, size)
    cv2.imwrite(f"{FILE}_processed.png", img)
