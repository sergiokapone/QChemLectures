from pyscf import gto, scf, cc
import numpy as np


def t1_diagnostic_survey(atoms_list, basis="cc-pvdz"):
    """
    T1 діагностика для серії атомів
    """

    print(f"\nT1 діагностика (базис: {basis})")
    print("=" * 70)
    print(f"{'Атом':6s} {'Спін':4s} {'T1 діаг.':12s} {'Оцінка':30s}")
    print("-" * 70)

    for symbol, spin in atoms_list:
        mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

        # Пропускаємо дуже великі атоми
        if mol.nelectron > 18:
            print(f"{symbol:6s} {spin:4d} (пропущено - занадто великий)")
            continue

        # HF
        if spin == 0:
            mf = scf.RHF(mol)
            mycc = cc.CCSD(mf)
        else:
            mf = scf.UHF(mol)
            mycc = cc.UCCSD(mf)

        mf.verbose = 0
        mf.conv_tol = 1e-10
        mf.kernel()

        # CCSD
        mycc.verbose = 0
        mycc.conv_tol = 1e-8

        try:
            e_corr, t1, t2 = mycc.kernel()

            # Обчислення T1 діагностики
            if spin == 0:
                t1_norm = np.linalg.norm(t1)
                n_elec = mol.nelectron
            else:
                t1_alpha, t1_beta = t1
                t1_norm = np.linalg.norm(t1_alpha)
                n_elec = mol.nelec[0]

            t1_diag = t1_norm / np.sqrt(n_elec)

            # Інтерпретація
            if t1_diag < 0.02:
                assessment = "Добре (одноконфігураційний)"
            elif t1_diag < 0.05:
                assessment = "Прийнятно (слабка МК)"
            else:
                assessment = "Погано (потрібен CASSCF)"

            print(f"{symbol:6s} {spin:4d} {t1_diag:12.6f} {assessment:30s}")

        except:
            print(f"{symbol:6s} {spin:4d} {'ПОМИЛКА':12s} {'Не конвергувало':30s}")

    print("=" * 70)
    print("\nЛегенда:")
    print("  T1 < 0.02: одноконфігураційний (CCSD надійний)")
    print("  0.02 < T1 < 0.05: слабка багатоконфігураційність")
    print("  T1 > 0.05: сильна МК (краще CASSCF/MRCI)")


# Тестування різних атомів
atoms_test = [
    ("He", 0),
    ("Be", 0),
    ("C", 2),
    ("N", 3),
    ("O", 2),
    ("F", 1),
    ("Ne", 0),
    ("Mg", 0),
]

t1_diagnostic_survey(atoms_test)
