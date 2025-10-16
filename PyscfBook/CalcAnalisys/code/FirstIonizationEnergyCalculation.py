from pyscf import gto, scf, dft
import numpy as np
import matplotlib.pyplot as plt


def calculate_ionization_energy(
    symbol, spin_neutral, spin_cation, method="HF", functional=None, basis="aug-cc-pvtz"
):
    """
    Розрахунок першої енергії іонізації

    Parameters:
    -----------
    symbol : str
        Символ атома
    spin_neutral : int
        2S для нейтрального атома
    spin_cation : int
        2S для катіона
    method : str
        'HF', 'MP2', 'CCSD', 'DFT'
    functional : str
        Функціонал для DFT (якщо method='DFT')
    basis : str
        Базисний набір
    """

    print(f"\nРозрахунок IE для {symbol} ({method}, {basis})")
    print("=" * 70)

    # Нейтральний атом
    mol_neutral = gto.M(
        atom=f"{symbol} 0 0 0", basis=basis, spin=spin_neutral, verbose=0
    )

    # Катіон
    mol_cation = gto.M(
        atom=f"{symbol} 0 0 0", basis=basis, charge=1, spin=spin_cation, verbose=0
    )

    # Розрахунок нейтрального атома
    if method == "HF":
        if spin_neutral == 0:
            mf_n = scf.RHF(mol_neutral)
        else:
            mf_n = scf.UHF(mol_neutral)
    elif method == "DFT":
        if spin_neutral == 0:
            mf_n = dft.RKS(mol_neutral)
        else:
            mf_n = dft.UKS(mol_neutral)
        mf_n.xc = functional
    elif method == "MP2":
        from pyscf import mp

        if spin_neutral == 0:
            mf_n = scf.RHF(mol_neutral)
        else:
            mf_n = scf.UHF(mol_neutral)
        mf_n.verbose = 0
        mf_n.kernel()
        mp_n = mp.MP2(mf_n) if spin_neutral == 0 else mp.UMP2(mf_n)
        mp_n.verbose = 0
        e_corr_n, _ = mp_n.kernel()
        e_neutral = mf_n.e_tot + e_corr_n
    elif method == "CCSD":
        from pyscf import cc

        if spin_neutral == 0:
            mf_n = scf.RHF(mol_neutral)
        else:
            mf_n = scf.UHF(mol_neutral)
        mf_n.verbose = 0
        mf_n.kernel()
        cc_n = cc.CCSD(mf_n) if spin_neutral == 0 else cc.UCCSD(mf_n)
        cc_n.verbose = 0
        e_corr_n, _, _ = cc_n.kernel()
        e_neutral = mf_n.e_tot + e_corr_n

    if method not in ["MP2", "CCSD"]:
        mf_n.verbose = 0
        mf_n.conv_tol = 1e-10
        e_neutral = mf_n.kernel()

    print(f"E({symbol}):  {e_neutral:.10f} Ha")

    # Розрахунок катіона
    if method == "HF":
        if spin_cation == 0:
            mf_c = scf.RHF(mol_cation)
        else:
            mf_c = scf.UHF(mol_cation)
    elif method == "DFT":
        if spin_cation == 0:
            mf_c = dft.RKS(mol_cation)
        else:
            mf_c = dft.UKS(mol_cation)
        mf_c.xc = functional
    elif method == "MP2":
        if spin_cation == 0:
            mf_c = scf.RHF(mol_cation)
        else:
            mf_c = scf.UHF(mol_cation)
        mf_c.verbose = 0
        mf_c.kernel()
        mp_c = mp.MP2(mf_c) if spin_cation == 0 else mp.UMP2(mf_c)
        mp_c.verbose = 0
        e_corr_c, _ = mp_c.kernel()
        e_cation = mf_c.e_tot + e_corr_c
    elif method == "CCSD":
        if spin_cation == 0:
            mf_c = scf.RHF(mol_cation)
        else:
            mf_c = scf.UHF(mol_cation)
        mf_c.verbose = 0
        mf_c.kernel()
        cc_c = cc.CCSD(mf_c) if spin_cation == 0 else cc.UCCSD(mf_c)
        cc_c.verbose = 0
        e_corr_c, _, _ = cc_c.kernel()
        e_cation = mf_c.e_tot + e_corr_c

    if method not in ["MP2", "CCSD"]:
        mf_c.verbose = 0
        mf_c.conv_tol = 1e-10
        e_cation = mf_c.kernel()

    print(f"E({symbol}+): {e_cation:.10f} Ha")

    # Енергія іонізації
    ie_ha = e_cation - e_neutral
    ie_ev = ie_ha * 27.211386

    print(f"\nIE({symbol}): {ie_ev:.4f} eV ({ie_ha * 1000:.4f} mHa)")

    return ie_ev, e_neutral, e_cation


# Приклади
ie_li = calculate_ionization_energy("Li", 1, 0, method="HF")
ie_c = calculate_ionization_energy("C", 2, 1, method="DFT", functional="pbe0")
ie_ne = calculate_ionization_energy("Ne", 0, 1, method="CCSD", basis="cc-pvdz")
