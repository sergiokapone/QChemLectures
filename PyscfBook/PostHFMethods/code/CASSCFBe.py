import numpy as np
from pyscf import gto, scf, mcscf


def casscf_beryllium_detailed(basis="cc-pvtz"):
    """
    Детальний CASSCF розрахунок для Be
    """

    mol = gto.M(atom="Be 0 0 0", basis=basis, spin=0, symmetry=True, verbose=0)

    print(f"\nCAS розрахунок атома Берилію (базис: {basis})")
    print("=" * 70)

    # Крок 1: HF розрахунок
    print("Крок 1: RHF розрахунок")
    mf = scf.RHF(mol)
    mf.verbose = 4
    mf.conv_tol = 1e-12
    e_hf = mf.kernel()

    print(f"\nRHF енергія: {e_hf:.10f} Ha")
    print(f"Електронна конфігурація HF: 1s² 2s²")

    # Крок 2: Вибір активного простору
    print("\nКрок 2: Визначення активного простору")
    print("Активний простір: CASSCF(4,8)")
    print("  4 електрони: валентні 2s² 2p⁰")
    print("  8 орбіталей: 2s, 2p (3 орб), 3s, 3p (3 орб)")

    # CASSCF(4,8): 4 електрони у 8 орбіталях
    ncas = 8  # активних орбіталей
    nelecas = 4  # активних електронів

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.verbose = 4
    mc.conv_tol = 1e-10

    # Крок 3: CASSCF розрахунок
    print("\nКрок 3: CASSCF оптимізація")
    e_casscf = mc.kernel()[0]

    print(f"\nCASCF(4,8) енергія: {e_casscf:.10f} Ha")
    print(f"Статична кореляція: {(e_casscf - e_hf) * 1000:.6f} mHa")

    # Аналіз хвильової функції
    print("\nКрок 4: Аналіз хвильової функції")

    # Природні орбіталі та заселеності
    natocc, natorb = mc.cas_natorb()

    print("\nЗаселеності природних орбіталей (активний простір):")
    for i, occ in enumerate(natocc):
        print(f"  Орбіталь {i + 1}: {occ:.6f}")

    # Аналіз конфігурацій
    print("\nАналіз CI коефіцієнтів:")

    # Для детального аналізу можна використати fcisolver
    ci = mc.ci

    # Вага основної конфігурації
    # (потребує додаткового аналізу CI вектора)

    # Entanglement entropy
    s_entropy = 0
    for occ in natocc:
        if occ > 1e-10 and occ < 2 - 1e-10:
            s_entropy += -occ / 2 * np.log(occ / 2) - (2 - occ) / 2 * np.log(
                (2 - occ) / 2
            )

    print(f"\nОднопартичкова ентропія: {s_entropy:.6f}")
    if s_entropy < 0.5:
        print("  → Слабка статична кореляція")
    elif s_entropy < 1.5:
        print("  → Помірна статична кореляція")
    else:
        print("  → Сильна статична кореляція")

    return e_hf, e_casscf, natocc


e_hf_be, e_cas_be, occ_be = casscf_beryllium_detailed()
