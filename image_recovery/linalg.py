# ================================

import numpy as np

# ================================
# Quaternion algebra
# ================================


def conjugate(Q: np.array) -> np.array:
    """
    Returns conjugate quaternion matrix represented as NumPy tensor of shape (*, *, 4).

    Parameters:
    ----------------
    Q: np.array
        tensor of shape (N, M, 4) representing quaternion matrix

    Returns:
    ----------------
    res: np.array
        tensor of shape (N, M, 4) representing quaternion matrix with imaginary part flipped

    Raises:
    ----------------
    ValueError:
        if tensor last axis has dimension different from 4
    """

    if Q.shape[2] != 4:
        raise ValueError("Wrong tensor shape")
    else:
        return np.concatenate([
            Q[:, :, :1], -Q[:, :, 1:]
        ], axis=2)


def frobenius_norm(Q: np.array) -> np.float64:
    """
    Returns Frobenius norm of quaternion matrix represented as NumPy tensor of shape (*, *, 4).

    Parameters:
    ----------------
    Q: np.array
        tensor of shape (N, M, 4) representing quaternion matrix

    Returns:
    ----------------
    res: np.float64
        Frobenius norm of supplied matrix

    Raises:
    ----------------
    ValueError:
        if tensor last axis has dimension different from 4
    """

    if Q.shape[2] != 4:
        raise ValueError("Wrong tensor shape")
    else:
        return np.sqrt(np.power(Q, 2).sum())


def qdot(Q1: np.array, Q2: np.array) -> np.array:
    """
    Performs multiplication of quaternion matrices represented as NumPy tensor of shape (*, *, 4).
    NOTE: VERY SLOW. FOR TESTING PURPOSES ONLY.

    Parameters:
    ----------------
    Q1, Q2: np.array
        3-dimensional tensors of shapes (N, M, 4), (M, K, 4) representing quaternion matrices

    Returns:
    ----------------
    res: np.array
        3-dimensional tensor of shape (N, K, 4) representing product of quaternion matrices

    Raises:
    ----------------
    ValueError:
        if tensors shapes are mismatched (Q1.shape[1] != Q2.shape[0]) or last axis has dimension different from 4

    """

    if (Q1.shape[2] != 4 or Q2.shape[2] != 4) or (Q1.shape[1] != Q2.shape[0]):
        raise ValueError("Wrong tensor shape or shapes mismatch")
    else:
        return np.stack([
            np.einsum("isk, sjk -> ij", Q1, Q2*np.array([1, -1, -1, -1])[None, None, :]),
            np.einsum("isk, sjk -> ij", Q1, Q2[:, :, (1, 0, 3, 2)]*np.array([1, 1, 1, -1])[None, None, :]),
            np.einsum("isk, sjk -> ij", Q1, Q2[:, :, (2, 3, 0, 1)]*np.array([1, -1, 1, 1])[None, None, :]),
            np.einsum("isk, sjk -> ij", Q1, Q2[:, :, (3, 2, 1, 0)]*np.array([1, 1, -1, 1])[None, None, :])
        ], axis=2)


# ================================
# Matrix transformations
# ================================


def qm2cm(Q: np.array) -> np.array:
    """
    Transforms quaternion matrix represented as NumPy tensor of shape (N, M, 4)
    to complex-valued matrix of shape (2N, 2M)
    based on mapping Q = Qa + Qb*j -> [[Qa, Qb], [-Qb*, Qa*]]

    Parameters:
    ----------------
    Q: np.array
        tensor of shape (N, M, 4) representing quaternion matrix

    Returns:
    ----------------
    res: np.array
        Corresponding 2-dimensional array of shape (2N, 2M) with complex elements

    Raises:
    ----------------
    ValueError:
        if tensor last axis has dimension different from 4
    """

    if Q.shape[2] != 4:
        raise ValueError("Wrong tensor shape or shapes mismatch")
    else:
        Qa = Q[:, :, 0] + 1j * Q[:, :, 1]
        Qb = Q[:, :, 2] + 1j * Q[:, :, 3]

        return np.vstack([
            np.hstack([Qa, Qb]),
            np.hstack([-np.conj(Qb), np.conj(Qa)])
        ])


def cm2qm(C: np.array) -> np.array:
    """
    Transforms complex-valued matrix of shape (2N, 2M)
    to quaternion matrix represented as NumPy tensor of shape (N, M, 4)
    based on mapping Q = Qa + Qb*j -> [[Qa, Qb], [-Qb*, Qa*]]

    Parameters:
    ----------------
    C: np.array
        complex-valued matrix of shape (2N, 2M) to transform

    Returns:
    ----------------
    res: np.array
        Corresponding 3-dimensional tensor of shape (N, M, 4) with real elements

    Raises:
    ----------------
    ValueError:
        if any of array axes has odd dimension
    """

    if (C.shape[0] % 2) or (C.shape[1] % 2):
        raise ValueError("Supplied matrix has odd shape")
    else:
        Qa = C[:C.shape[0] // 2, :C.shape[1] // 2]
        Qb = -C[:C.shape[0] // 2, C.shape[1] // 2:]

        return np.stack([
            np.real(Qa), np.imag(Qa), np.real(Qb), np.imag(Qb)
        ], axis=2)
