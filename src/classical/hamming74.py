"""Code correcteur de Hamming (7,4) à décisions dures.

Chaque groupe de 4 bits d'information donne 7 bits transmis.
Un seul bit erroné par bloc de 7 peut être corrigé. Deux erreurs ou
plus dans le même bloc ne sont pas garanties corrigibles.
"""

import numpy as np


def encode_hamming74(bits: np.ndarray) -> tuple[np.ndarray, int]:
    """Encode en Hamming (7,4); retourne (bits_codes, nb_bits_remplissage)."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("Les entrées doivent être des bits 0 ou 1.")
    padding = (-bits.size) % 4
    if padding:
        bits = np.pad(bits, (0, padding))
    d = bits.reshape(-1, 4)
    d1, d2, d3, d4 = d.T
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p4 = d2 ^ d3 ^ d4
    code = np.stack((p1, p2, d1, p4, d2, d3, d4), axis=1)
    return code.reshape(-1), padding


def decode_hamming74(bits_codes: np.ndarray, padding: int = 0) -> tuple[np.ndarray, int]:
    """Décode Hamming (7,4) et renvoie (bits_recuperes, syndromes_non_nuls).

    Les syndromes non nuls déclenchent une correction supposant une erreur
    unique. Ce compteur n'est PAS une preuve du nombre d'erreurs corrigées.
    """
    bits_codes = np.asarray(bits_codes, dtype=np.uint8).reshape(-1)
    if bits_codes.size % 7:
        raise ValueError("Le nombre de bits codés doit être multiple de 7.")
    if not 0 <= padding <= 3:
        raise ValueError("padding doit être compris entre 0 et 3.")
    if np.any((bits_codes != 0) & (bits_codes != 1)):
        raise ValueError("Les entrées doivent être des bits 0 ou 1.")
    r = bits_codes.reshape(-1, 7).copy()
    s1 = r[:, 0] ^ r[:, 2] ^ r[:, 4] ^ r[:, 6]
    s2 = r[:, 1] ^ r[:, 2] ^ r[:, 5] ^ r[:, 6]
    s4 = r[:, 3] ^ r[:, 4] ^ r[:, 5] ^ r[:, 6]
    syndrome = s1 + 2 * s2 + 4 * s4
    faulty = np.flatnonzero(syndrome)
    r[faulty, syndrome[faulty].astype(np.int64) - 1] ^= 1
    decoded = r[:, [2, 4, 5, 6]].reshape(-1)
    if padding:
        decoded = decoded[:-padding]
    return decoded, int(faulty.size)
