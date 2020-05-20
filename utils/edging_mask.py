from scipy import ndimage
import numpy as np


def edging_mask(orig_input):
    erode_arr = []
    for i in range(orig_input.shape[-1]):
        orig_slice = orig_input[:,:,i]
        struct = ndimage.generate_binary_structure(2, 2)
        erode_slice = ndimage.binary_erosion(orig_slice, struct)
        erode_arr.append(erode_slice)
    erode_input = np.stack(erode_arr, axis=-1)

    edge_input = orig_input - erode_input

    return edge_input