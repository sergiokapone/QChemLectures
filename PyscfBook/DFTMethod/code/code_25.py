from pyscf import gto, scf, dft


def electron_affinity_comparison(atoms_data, basis="aug-cc-pvqz"):
    """
    Порівняння електронної спорідненості
    EA = E(A) - E(A⁻)
    """

    methods = ["HF", "LDA", "PBE", "B3LYP", "PBE0"]

    print(f"\nЕлектронна спорідненість (eV), базис: {basis}")
    print("=" * 90)
    print(
        f"{'Атом':6s} {'Експ.':10s} {'HF':10s} {'LDA':10s} "
        f"{'PBE':10s} {'B3LYP':10s} {'PBE0':10s}"
    )
    print("-" * 90)

    for symbol, spin_n, spin_a, ea_exp in atoms_data:
        # Нейтральний атом
        mol_n = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin_n, verbose=0)

        # Аніон
        mol_a = gto.M(
            atom=f"{symbol} 0 0 0", basis=basis, charge=-1, spin=spin_a, verbose=0
        )

        ea_values = [ea_exp]

        for method in methods:
            if method == "HF":
                mf_n = scf.UHF(mol_n) if spin_n > 0 else scf.RHF(mol_n)
                mf_a = scf.UHF(mol_a) if spin_a > 0 else scf.RHF(mol_a)
            else:
                mf_n = dft.UKS(mol_n) if spin_n > 0 else dft.RKS(mol_n)
                mf_a = dft.UKS(mol_a) if spin_a > 0 else dft.RKS(mol_a)

                xc_dict = {"LDA": "lda", "PBE": "pbe", "B3LYP": "b3lyp", "PBE0": "pbe0"}
                mf_n.xc = xc_dict[method]
                mf_a.xc = xc_dict[method]

            mf_n.verbose = 0
            mf_a.verbose = 0
            mf_a.level_shift = 0.5  # Для аніонів часто потрібно

            try:
                e_n = mf_n.kernel()
                e_a = mf_a.kernel()

                ea = (e_n - e_a) * 27.211386  # eV
                ea_values.append(ea)
            except:
                ea_values.append(np.nan)

        # Виведення
        row_str = f"{symbol:6s}"
        for ea in ea_values:
            if not np.isnan(ea):
                row_str += f" {ea:10.4f}"
            else:
                row_str += f" {'N/A':10s}"
        print(row_str)

    print("=" * 90)
    print("\nПримітка: EA > 0 означає, що аніон стабільний")


# Дані: (символ, спін нейтрального, спін аніона, EA експ.)
atoms_ea = [
    ("B", 1, 2, 0.28),
    ("C", 2, 3, 1.26),
    ("O", 2, 1, 1.46),
    ("F", 1, 0, 3.40),
    ("Cl", 1, 0, 3.61),
    ("Br", 1, 0, 3.36),
]

electron_affinity_comparison(atoms_ea)
