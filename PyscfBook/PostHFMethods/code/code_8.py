from pyscf import gto, scf, cc
import numpy as np


def ccsd_helium_detailed(basis="cc-pvqz"):
    """
    Детальний CCSD розрахунок для He
    """

    mol = gto.M(atom="He 0 0 0", basis=basis, spin=0, verbose=0)

    print(f"\nCCSD розрахунок атома Гелію (базис: {basis})")
    print("=" * 70)

    # Крок 1: HF
    print("\nКрок 1: Hartree-Fock")
    mf = scf.RHF(mol)
    mf.verbose = 4
    mf.conv_tol = 1e-12
    e_hf = mf.kernel()

    print(f"\nHF енергія: {e_hf:.12f} Ha")

    # Крок 2: CCSD
    print("\nКрок 2: CCSD розрахунок")
    mycc = cc.CCSD(mf)
    mycc.verbose = 4
    mycc.conv_tol = 1e-10

    e_ccsd_corr, t1, t2 = mycc.kernel()
    e_ccsd = e_hf + e_ccsd_corr

    print(f"\nCCSD кореляція: {e_ccsd_corr:.12f} Ha")
    print(f"CCSD повна енергія: {e_ccsd:.12f} Ha")

    # Аналіз амплітуд
    print(f"\nАналіз амплітуд:")
    print(f"  T1 амплітуди: {t1.shape}")
    print(f"  T2 амплітуди: {t2.shape}")
    print(f"  |T1|_max = {np.max(np.abs(t1)):.6f}")
    print(f"  |T2|_max = {np.max(np.abs(t2)):.6f}")
    print(f"  ||T1||_2 = {np.linalg.norm(t1):.6f}")
    print(f"  ||T2||_2 = {np.linalg.norm(t2):.6f}")

    # Для He T1 має бути дуже малим (симетрія)
    if np.max(np.abs(t1)) < 1e-6:
        print("  T1 ≈ 0 (як і очікувалось для замкненої оболонки)")

    # Крок 3: CCSD(T)
    print("\nКрок 3: (T) корекція")
    e_t = mycc.ccsd_t()
    e_ccsdt = e_ccsd + e_t

    print(f"\n(T) корекція: {e_t:.12f} Ha")
    print(f"CCSD(T) повна енергія: {e_ccsdt:.12f} Ha")

    # Порівняння з експериментом
    e_exp = -2.903724377  # Ha (високоточне значення)

    print(f"\nПорівняння з експериментом:")
    print(f"Експеримент: {e_exp:.12f} Ha")
    print(f"HF похибка:     {(e_hf - e_exp) * 1000:10.6f} mHa")
    print(f"CCSD похибка:   {(e_ccsd - e_exp) * 1000:10.6f} mHa")
    print(f"CCSD(T) похибка:{(e_ccsdt - e_exp) * 1000:10.6f} mHa")

    # Розбиття кореляції
    print(f"\nРозбиття кореляційної енергії:")
    total_corr = e_ccsdt - e_hf
    print(f"Повна кореляція: {total_corr * 1000:.6f} mHa (100%)")
    print(
        f"CCSD внесок:     {e_ccsd_corr * 1000:.6f} mHa "
        f"({abs(e_ccsd_corr / total_corr) * 100:.2f}%)"
    )
    print(f"(T) внесок:      {e_t * 1000:.6f} mHa ({abs(e_t / total_corr) * 100:.2f}%)")

    return e_hf, e_ccsd, e_ccsdt


e_hf_he, e_ccsd_he, e_ccsdt_he = ccsd_helium_detailed()
