from pyscf import gto, scf, mp, cc, ci
import numpy as np


def correlation_energy_demo(symbol, spin, basis="cc-pvdz"):
    """
    Демонстрація кореляційної енергії
    """

    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

    print(f"\nКореляційна енергія атома {symbol} (базис: {basis})")
    print("=" * 70)

    # HF розрахунок
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)

    mf.verbose = 0
    mf.conv_tol = 1e-10
    e_hf = mf.kernel()

    print(f"HF енергія:           {e_hf:.8f} Ha")

    # MP2
    if spin == 0:
        mp2 = mp.MP2(mf)
    else:
        mp2 = mp.UMP2(mf)

    mp2.verbose = 0
    e_mp2, t2 = mp2.kernel()
    e_total_mp2 = e_hf + e_mp2

    print(f"MP2 кореляція:        {e_mp2:.8f} Ha")
    print(f"MP2 повна енергія:    {e_total_mp2:.8f} Ha")

    # CCSD (якщо атом не дуже великий)
    if mol.nelectron <= 10:
        if spin == 0:
            mycc = cc.CCSD(mf)
        else:
            mycc = cc.UCCSD(mf)

        mycc.verbose = 0
        e_ccsd, t1, t2 = mycc.kernel()
        e_total_ccsd = e_hf + e_ccsd

        print(f"CCSD кореляція:       {e_ccsd:.8f} Ha")
        print(f"CCSD повна енергія:   {e_total_ccsd:.8f} Ha")

        # CCSD(T)
        e_t = mycc.ccsd_t()
        e_total_ccsdt = e_total_ccsd + e_t

        print(f"(T) корекція:         {e_t:.8f} Ha")
        print(f"CCSD(T) повна енергія:{e_total_ccsdt:.8f} Ha")
    else:
        print("CCSD пропущено (атом занадто великий для демо)")

    print("=" * 70)

    # Аналіз
    print(f"\nАналіз:")
    print(f"Кореляція складає {abs(e_mp2 / e_hf) * 100:.2f}% від HF енергії")

    if mol.nelectron <= 10:
        print(f"MP2 відновлює {abs(e_mp2 / e_ccsd) * 100:.1f}% CCSD кореляції")


# Приклади
correlation_energy_demo("He", spin=0)
correlation_energy_demo("Be", spin=0)
correlation_energy_demo("C", spin=2)
correlation_energy_demo("Ne", spin=0)
