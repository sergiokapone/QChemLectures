import numpy as np

def transform_to_normal_coordinates(hessian, normal_modes):
    """
    Перетворює гессіан у нормальні координати.

    :param hessian: numpy.ndarray, матриця Гессе (NxN)
    :param normal_modes: numpy.ndarray, матриця нормальних мод (NxN), стовпці - нормальні координати
    :return: (diagonal_hessian, frequencies_cm1)
    """
    # Преобразуем гессиан: Q^T H Q
    H_diag = normal_modes.T @ hessian @ normal_modes

    # Частоти в атомних одиницях: корінь із діагональних елементів
    omega_squared = np.diag(H_diag)
    omega_au = np.sqrt(np.abs(omega_squared)) * np.sign(omega_squared)  # враховуємо негативні значення

    # Conversion to cm-¹ (1 a.u. of frequency ≈ 5140.48 cm-¹)
    conversion_factor = 5140.48
    frequencies_cm1 = omega_au * conversion_factor

    return H_diag, frequencies_cm1


if __name__ == "__main__":
    # hessian - 6x6 numpy array (with ORCA)
    hessian = np.array([
    [ 4.0174144836E-01,  6.2401727647E-20,  2.2369839367E-20, -4.0174144836E-01, -6.3488478200E-20, -2.2008339933E-20],
    [ 6.2401727647E-20, -1.9756629140E-06,  8.1315162936E-19, -6.2401727647E-20,  1.9756629160E-06, -6.5052130349E-19],
    [ 2.2369839367E-20,  8.1315162936E-19, -1.9756629140E-06, -2.2369839367E-20, -2.1684043450E-19,  1.9756629160E-06],
    [-4.0174144836E-01, -6.2401727647E-20, -2.2369839367E-20,  4.0174144836E-01,  6.3488478200E-20,  2.2008339933E-20],
    [-6.3488478200E-20,  1.9756629160E-06, -2.1684043450E-19,  6.3488478200E-20, -1.9756629138E-06,  1.0842021725E-19],
    [-2.2008339933E-20, -6.5052130349E-19,  1.9756629160E-06,  2.2008339933E-20,  1.0842021725E-19, -1.9756629138E-06]
    ])
    # normal_modes - 6x6 numpy array (from $normal_modes; each column is a mode)
    normal_modes = np.array([
    [ 0.0000000000E+00, 0, 0, 0, 0,  7.0710678119E-01],
    [ 0.0000000000E+00, 0, 0, 0, 0,  0.0000000000E+00],
    [ 0.0000000000E+00, 0, 0, 0, 0,  0.0000000000E+00],
    [ 0.0000000000E+00, 0, 0, 0, 0, -7.0710678119E-01],
    [ 0.0000000000E+00, 0, 0, 0, 0,  0.0000000000E+00],
    [ 0.0000000000E+00, 0, 0, 0, 0,  0.0000000000E+00]
        ])

    H_diag, freqs = transform_to_normal_coordinates(hessian, normal_modes)

    print("Діагональний гессиан:")
    print(np.round(H_diag, 6))  # округлення для зручності

    print("Частоти (см⁻¹):")
    print(np.round(freqs, 2))
