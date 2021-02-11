import numpy as np
import scipy
def s2t(x):
    """Convert [0, 1] float to [-1, 1] float
    """
    return x*2-1

def u2t(x):
    """Convert uint8 to [-1, 1] float
    """
    return x.astype('float32') / 255*2-1

def resize_to_32(x):
    
    resized_x = np.empty((len(x), 32, 32, 3), dtype='float32')
    for i, img in enumerate(x):
        # imresize returns uint8
        resized_x[i] = scipy.misc.imresize(img, (32, 32))
    return resized_x