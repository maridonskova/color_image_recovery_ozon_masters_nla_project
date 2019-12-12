# ================================

import numpy as np
import cv2

from itertools import product

# ================================


def img2qm(filepath: str) -> np.array:
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB).astype(np.float64)/255.0

    return np.concatenate([
        np.zeros((*img.shape[:2], 1)),
        img[:, :, ]
    ], axis=2)


# ================================


def add_random_missing_pixels(img: np.array, q: float, mode: str = "uniform") -> (np.array, np.array):
    mask = np.ones(img.shape[:2], dtype=np.bool)

    if mode == "uniform":
        idxs = np.random.choice(np.prod(img.shape[:2]), size=int(np.prod(img.shape[:2])*q), replace=False)
        mask[[x // img.shape[1] for x in idxs], [x % img.shape[1] for x in idxs]] = False
    elif mode == "square":
        sqsz = int(np.sqrt(np.prod(img.shape[:2])*q))
        startpos = (np.random.choice(img.shape[0] - sqsz), np.random.choice(img.shape[1] - sqsz))
        mask[startpos[0]:(startpos[0] + sqsz), startpos[1]:(startpos[1] + sqsz)] = False

    imgx = img.copy()
    imgx[np.tile(~mask[:, :, None], (1, 1, 4))] = 0.0

    return imgx, mask
