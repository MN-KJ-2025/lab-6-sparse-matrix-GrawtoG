# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import scipy as sp


def is_diagonally_dominant(A: np.ndarray | sp.sparse.csc_array) -> bool | None:
    """Funkcja sprawdzająca czy podana macierz jest diagonalnie zdominowana.

    Args:
        A (np.ndarray | sp.sparse.csc_array): Macierz A (m,m) podlegająca
            weryfikacji.

    Returns:
        (bool): `True`, jeśli macierz jest diagonalnie zdominowana,
            w przeciwnym wypadku `False`.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    try:
        if isinstance(A, sp.sparse._csc.csc_array):
            A = A.toarray()
        if not isinstance(A, (np.ndarray, sp.sparse.csc_array)):
            return None
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return None

        abs_row_sums = np.sum(np.abs(A), axis=1)
        diag_abs = np.abs(np.diagonal(A))
        offdiag = abs_row_sums - diag_abs
        return np.all(diag_abs > offdiag)
    except Exception:
        return None


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float | None:
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b.

    Args:
        A (np.ndarray): Macierz A (m,n) zawierająca współczynniki równania.
        x (np.ndarray): Wektor x (n,) zawierający rozwiązania równania.
        b (np.ndarray): Wektor b (m,) zawierający współczynniki po prawej
            stronie równania.

    Returns:
        (float): Wartość normy residuum dla podanych parametrów.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    try:
        if isinstance(A, sp.sparse._csc.csc_array):
            A = A.toarray()
        if not (
            isinstance(A, np.ndarray)
            and isinstance(x, np.ndarray)
            and isinstance(b, np.ndarray)
        ):
            return None

        if A.shape[0] != b.shape[0]:
            return None
        if (A.shape)[1] != (x.shape)[0]:
            return None

        if A.ndim != 2 or b.ndim != 1 or x.ndim != 1:
            return None

        return np.linalg.norm(b - (A @ x))
    except Exception:
        return None
