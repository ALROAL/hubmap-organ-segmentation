import numpy as np
import cv2
import tifffile

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    img = img.reshape(shape)
    mask_corrected = cv2.flip(cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE),1)
    return mask_corrected  # Needed to align to RLE direction


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# Modified from https://www.kaggle.com/code/abhinand05/hubmap-extensive-eda-what-are-we-hacking
def read_tiff(path, scale=None,
              verbose=0):
    image = tifffile.imread(path)
    if len(image.shape) == 5:
        image = image.squeeze().transpose(1, 2, 0)

    if verbose:
        print(f"[{path}] Image shape: {image.shape}")

    if scale:
        new_size = (image.shape[1] // scale, image.shape[0] // scale)
        image = cv2.resize(image, new_size)

        if verbose:
            print(f"[{path}] Resized Image shape: {image.shape}")

    #     mx = np.max(image)
    #     image = image.astype(np.float32)
    #     if mx:
    #         image /= mx # scale image to [0, 1]
    return image


def read_img(path):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    return image

def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return mask