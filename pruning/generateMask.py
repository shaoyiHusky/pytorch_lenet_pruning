import numpy as np


def lenet_prune():
    # generate mask
    masks = []
    masks.append(np.array([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]],
                           [[[0, 1, 0], [1, 1, 1], [0, 1, 0]]],
                           [[[0, 1, 0], [1, 1, 1], [0, 1, 0]]],
                           [[[0, 1, 0], [1, 1, 1], [0, 1, 0]]],
                           [[[0, 1, 0], [1, 1, 1], [0, 1, 0]]],
                           [[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]]))
    # print(masks[0].shape)
    return masks
