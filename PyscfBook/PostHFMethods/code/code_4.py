from pyscf import gto, scf, mp
import numpy as np


def mp2_neon_calculation(basis="cc-pvtz"):
    """
    Детальний MP2 розрахунок атома Ne
    """

    mol = gto.M(atom="Ne 0 0 0", basis=basis, spin=0, verbose=0)

    print(f"\nMP2 розрахунок атома Неону (базис: {basis})")
    print("=" * 70)

    # Крок 1: HF розрахунок
    print("\nКрок 1: Hartree-Fock розрахунок")
    mf = scf.RHF(mol)
    mf.verbose = 4
    mf.conv_tol = 1e-12
    e_hf = mf.kernel()

    print(f"\nHF енергія: {e_hf:.10f} Ha")
    print(f"HF енергія: {e_hf * 27.211386:.6f} eV")

    # Крок 2: MP2 розрахунок
    print("\nКрок 2: MP2 кореляція")
    mymp2 = mp.MP2(mf)
    mymp2.verbose = 4
    e_mp2_corr, t2 = mymp2.kernel()

    e_total_mp2 = e_hf + e_mp2_corr

    print(f"\nMP2 кореляційна енергія: {e_mp2_corr:.10f} Ha")
    print(f"MP2 повна енергія: {e_total_mp2:.10f} Ha")
    print(f"MP2 повна енергія: {e_total_mp2 * 27.211386:.6f} eV")

    # Аналіз
    print("\nАналіз:")
    print(f"Кореляція становить {abs(e_mp2_corr / e_hf) * 100:.3f}% від HF енергії")

    # Порівняння з експериментом
    e_exp = -128.937  # Ha (експериментальна енергія Ne)
    error_hf = (e_hf - e_exp) * 1000
    error_mp2 = (e_total_mp2 - e_exp) * 1000

    print(f"\nПорівняння з експериментом ({e_exp:.6f} Ha):")
    print(f"  HF похибка:  {error_hf:8.4f} mHa")
    print(f"  MP2 похибка: {error_mp2:8.4f} mHa")
    print(f"  Покращення: {abs(error_hf - error_mp2):8.4f} mHa")
    print(f"  MP2 відновлює {(1 - error_mp2 / error_hf) * 100:.1f}% похибки HF")

    # Інформація про розміри
    print(f"\nОбчислювальні деталі:")
    print(f"  Заповнених орбіталей: {mol.nelectron // 2}")
    print(f"  Віртуальних орбіталей: {mol.nao_nr() - mol.nelectron // 2}")
    print(f"  Розмір t2 амплітуд: {t2.shape}")

    return e_hf, e_total_mp2, e_mp2_corr


e_hf, e_mp2, e_corr = mp2_neon_calculation()
