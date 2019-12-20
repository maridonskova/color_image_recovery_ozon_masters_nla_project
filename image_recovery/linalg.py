# ================================

import numpy as np
import scipy.linalg as splin

from tqdm.notebook import tqdm

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


def qm2cm(Q: np.array, mask: np.array = None) -> np.array:
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

        C = np.vstack([
            np.hstack([Qa, Qb]),
            np.hstack([-np.conj(Qb), np.conj(Qa)])
        ])

        if mask is None:
            C_mask = None
        else:
            C_mask = np.tile(mask, (2, 2))

        return C, C_mask


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
        Qb = C[:C.shape[0] // 2, C.shape[1] // 2:]

        return np.stack([
            np.real(Qa), np.imag(Qa), np.real(Qb), np.imag(Qb)
        ], axis=2)


# ================================
# Matrix recovery
# ================================


def lrqmc(mtr: np.array, mask: np.array, init_rank: int = None, min_rank: int = 2,
          reg_coef: float = 1e-3, max_iter: int = 100, rel_tol: float = 1e-3,
          hard_rank_reduction: bool = True, rot: float = 10.0, rank_mult: float = 0.9,
          full_history=False, random_state=None, progress: bool = True):
    """
    LRQMC method of restoring color image with some of pixels missing

    Parameters:
    ----------------
    mtr: np.array
        Image to be restored, represented either as 3-tensor of shape (N, M, 4) or matrix of shape (N, M) - greyscale.
        This is determined by dimensions of mtr.
        Color channels are: 1 - red, 2 - blue, 3 - green
    mask: np.array
        Boolean mask of shape (N, M) signaling missing pixels.
        Entries with False correspond to missing values
    init_rank: int
        Initial estimation of low-rank approximation. Must be in [2, min(2N, 2M)]
    min_rank: int
        Minimal possible rank of UV-decomposition of quaternion matrix
    reg_coef: float > 0.0
        Regularization coefficient
    max_iter: int
        Maximum allowable number of iterations.
        If required tolerance not achieved after max_iter iterations - computations end
    progress: int > 0
        Controls how often to print output info.
        If zero - no info is printed, if n - info is printed every nth iteration
    rel_tol: float > 0.0
        Convergence tolerance.
        When norm(X[i + 1] - X[i])/X[i] < rel_tol - convergence is achieved (X is restored image)
    hard_rank_reduction: bool
        If True, then rank reduction procedure is acting hard, meaning that it reduces rank to r,
        where r maximizes eigv[r] / eigv[r + 1] - quotient of eigenvalues of U*U^H.
        Otherwise rank is multiplied by rank_mult
    rot: float > 0.0
        Rank Overestimation Threshold.
        Controls when to reduce rank estimation. Rank is reduced when eigv[max]*(rank - 1)/sum(eigv) > rot,
        where eigv - eigenvalues of U*U^H, rank - current estimation of rank
    rank_mult: float in (0.0, 1.0)
        Rank Multiplier.
        When Rank Overestimation Threshold is exceeded, rank estimation is multiplied by this number
    return_norms: bool
        If True, function returns norms sequence as well.
    random_state: int
        NumPy random generator seed

    Returns:
    ----------------
    Q: np.array
        Restored image, represented as 3-tensor of shape (N, M, 4).
    U, V: np.arrays
        Multiplicants of Q: C(Q) = UV, where C is transformation from quaternion matrix to complex matrix
    norms: list of floats
        Sequence of norms of restored images per iteration
    """

    if mtr.ndim == 3:
        X, mask = qm2cm(mtr, mask.copy())
    else:
        X, mask = mtr.copy(), mask.copy()

    X0 = X.copy()

    # ================================

    if (not init_rank) or (init_rank > min(X.shape)):
        c_rank = min(X.shape)
    else:
        c_rank = init_rank

    np.random.seed(random_state)
    U = np.random.uniform(size=(X.shape[0], c_rank)) + 1j*np.random.uniform(size=(X.shape[0], c_rank))
    V = np.random.uniform(size=(c_rank, X.shape[1])) + 1j*np.random.uniform(size=(c_rank, X.shape[1]))

    norms = np.zeros(max_iter + 1, dtype=np.float64)
    norms[0] = np.linalg.norm(X - U.dot(V))

    # ================================

    mu = 0.0

    flag = True
    ix = 0

    if full_history:
        fh = [mtr]

    if progress:
        pbar = tqdm(total=max_iter, leave=False, desc="LRQMC", postfix=f"Initial rank estimation: {c_rank}.")

    while flag:
        U = (X.dot(np.conj(V.T))).dot(splin.pinv(V.dot(np.conj(V.T)) + reg_coef*np.eye(c_rank), return_rank=False))
        V = splin.pinv(np.conj(U.T).dot(U) + reg_coef*np.eye(c_rank), return_rank=False).dot((np.conj(U.T)).dot(X))
        X = X0.copy()
        X[~mask] += (U.dot(V))[~mask]

        # ================================

        if c_rank > min_rank:
            eigvs = np.sort(splin.eigvalsh(np.conj(U.T).dot(U)))[::-1]
            quots = eigvs[:-1]/eigvs[1:]
            max_ind = np.argmax(quots)
            mu = (c_rank - 1.0)/(quots.sum()/quots[max_ind] - 1.0)

            if mu > rot:
                if hard_rank_reduction:
                    c_rank = max(max_ind + 1, min_rank)
                else:
                    c_rank = max(int(rank_mult*c_rank), min_rank)

                XU, XS, XVh = splin.svd(X, compute_uv=True, full_matrices=False)
                U = XU[:, :c_rank]*XS[None, :c_rank]
                V = XVh[:c_rank, :]

                X = X0.copy()
                X[~mask] += (U.dot(V))[~mask]

        # ================================

        norms[ix + 1] = np.linalg.norm(X - U.dot(V))
        rel_norm_change = (norms[ix + 1] - norms[ix])/norms[ix]

        if full_history:
            fh.append(cm2qm(X))

        # ================================

        if progress:
            pbar.update()
            pbar.set_postfix_str(f"Norm changed: {rel_norm_change*100.0:.3f} %. "
                                 f"Rank / overestimation: {c_rank} / {mu:.2f}")

        if -rel_tol < rel_norm_change < 0.0:
            if progress:
                pbar.close()
                print(f"Iteration {ix + 1}. Required relative tolerance achieved")

            flag = False
        elif ix >= max_iter - 1:
            if progress:
                pbar.close()
                print(f"Max iterations count achieved.")

            flag = False
        else:
            ix += 1

    # ================================

    if mtr.ndim == 3:
        X = cm2qm(X)

    if full_history:
        return fh
    else:
        return X, U, V


