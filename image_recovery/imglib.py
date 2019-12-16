# ================================

import numpy as np
import cv2

from itertools import product

# ================================


def img2qm(filepath: str) -> np.array:
    """
    Reads image from file in filesystem and transforms it to NumPy 3-tensor

    Parameters:
    ----------------
    filepath: str
        Path to file

    Returns:
    ----------------
        img: np.array
            3-tensor representing the image. Last axis has dimension 4, and contains color channels: (0, R, G, B)
    """

    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB).astype(np.float64)/255.0

    return np.concatenate([
        np.zeros((*img.shape[:2], 1)),
        img[:, :, ]
    ], axis=2)


# ================================


def add_random_missing_pixels(img: np.array, q: float, mode: str = "uniform",
                              random_state: int = None, **kwargs) -> (np.array, np.array):
    """
        Randomly removes pixels from picture in selected fashion, returning altered picture and
        boolean mask

        Parameters:
        ----------------
        img: np.array
            3-tensor representing the image
        q: 0.0 <= float <= 1.0
            proportion of pixels to erase
        mode: str
            Pixels removal fashion. Should be one of:
                uniform: each pixel has equal probability to get erased
                square: erases square randomly positioned within picture. Square area is q*(picure area)

        Returns:
        ----------------
        imgx: np.array
            3-tensor representing the image with pixels erased
        mask: np.array
            boolean 2-array, True values correspond to pixels that have not been erased

        Raises:
        ----------------
        ValueError:
            if q not in [0.0, 1.0] or tensor last axis has dimension different from 4
    """

    if not ((0.0 <= q <= 1.0) or (img.shape[2] != 4)):
        raise ValueError("Wrong tensor shape or erased pixels proportion not in [0, 1]")
    else:
        mask = np.zeros(img.shape[:2], dtype=np.bool)
        np.random.seed(random_state)

        if mode == "uniform":
            idxs = np.random.choice(np.prod(img.shape[:2]), size=int(np.prod(img.shape[:2])*q), replace=False)
            mask[[x // img.shape[1] for x in idxs], [x % img.shape[1] for x in idxs]] = True

        # ================================

        elif mode == "normal_clusters":
            n_clusters = kwargs.get("n_clusters", 3)
            stdd = kwargs.get("std", min(img.shape[:2])/10)
            max_tries = kwargs.get("max_tries", 10)

            cluster_centers = np.array([
                np.random.randint(img.shape[0], size=n_clusters),
                np.random.randint(img.shape[1], size=n_clusters)
            ]).T

            pix_prop = 0.0
            tries = 0

            while (tries < max_tries) and (pix_prop < q):
                new_pixs = np.concatenate([
                    np.random.multivariate_normal(xcl, np.eye(2) * stdd ** 2,
                                                  size=int(np.ceil(img.size * 0.1 / n_clusters)))
                    for xcl in cluster_centers
                ]).astype(np.int32)

                new_pixs[:, 0] = np.clip(new_pixs[:, 0], 0, img.shape[0] - 1)
                new_pixs[:, 1] = np.clip(new_pixs[:, 1], 0, img.shape[1] - 1)
                mask[new_pixs[:, 0], new_pixs[:, 1]] = True
                pix_prop = mask.sum()/mask.size

                tries += 1

        # ================================

        elif mode == "square":
            sqsz = int(np.sqrt(np.prod(img.shape[:2])*q))
            startpos = (np.random.choice(img.shape[0] - sqsz), np.random.choice(img.shape[1] - sqsz))
            mask[startpos[0]:(startpos[0] + sqsz), startpos[1]:(startpos[1] + sqsz)] = True
        else:
            raise ValueError(f"Unknown option {mode}")

        imgx = img.copy()
        imgx[np.tile(mask[:, :, None], (1, 1, 4))] = 0.0

        return imgx, ~mask
