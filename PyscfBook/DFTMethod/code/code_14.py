from pyscf import gto, scf, dft


def analyze_functional_effect(symbol, charge, spin_n, spin_c, basis="aug-cc-pvtz"):
    """
    Аналіз впливу функціоналу на енергію іонізації
    """

    functionals = ["pbe", "blyp", "b3lyp", "pbe0", "cam-b3lyp"]

    print(f"\nЕнергії іонізації {symbol} (базис: {basis})")
    print("=" * 70)
    print(f"{'Метод':15s} {'E(A), Ha':15s} {'E(A+), Ha':15s} {'IE, eV':10s}")
    print("-" * 70)

    # HF референс
    mol_n = gto.M(
        atom=f"{symbol} 0 0 0", basis=basis, charge=charge, spin=spin_n, verbose=0
    )
    mol_c = gto.M(
        atom=f"{symbol} 0 0 0", basis=basis, charge=charge + 1, spin=spin_c, verbose=0
    )

    if spin_n == 0:
        mf_n = scf.RHF(mol_n)
    else:
        mf_n = scf.UHF(mol_n)

    if spin_c == 0:
        mf_c = scf.RHF(mol_c)
    else:
        mf_c = scf.UHF(mol_c)

    mf_n.verbose = 0
    mf_c.verbose = 0

    e_n_hf = mf_n.kernel()
    e_c_hf = mf_c.kernel()
    ie_hf = (e_c_hf - e_n_hf) * 27.211386

    print(f"{'HF':15s} {e_n_hf:15.8f} {e_c_hf:15.8f} {ie_hf:10.4f}")

    # DFT функціонали
    for xc in functionals:
        if spin_n == 0:
            mf_n = dft.RKS(mol_n)
        else:
            mf_n = dft.UKS(mol_n)

        if spin_c == 0:
            mf_c = dft.RKS(mol_c)
        else:
            mf_c = dft.UKS(mol_c)

        mf_n.xc = xc
        mf_c.xc = xc
        mf_n.verbose = 0
        mf_c.verbose = 0

        try:
            e_n = mf_n.kernel()
            e_c = mf_c.kernel()
            ie = (e_c - e_n) * 27.211386

            print(f"{xc.upper():15s} {e_n:15.8f} {e_c:15.8f} {ie:10.4f}")
        except:
            print(f"{xc.upper():15s} помилка розрахунку")

    print("=" * 70)


# Приклади
analyze_functional_effect("C", 0, 2, 1)  # C → C+
analyze_functional_effect("O", 0, 2, 3)  # O → O+
analyze_functional_effect("Ne", 0, 0, 1)  # Ne → Ne+
