from pyscf import gto, scf, dft


def spin_contamination_analysis(atoms_list, basis="cc-pvtz"):
    """
    Порівняння забруднення спіном у HF та DFT
    """

    print(f"\nЗабруднення спіном <S²> (базис: {basis})")
    print("=" * 80)
    print(
        f"{'Атом':6s} {'2S':4s} {'<S²> очік.':12s} {'HF':12s} "
        f"{'LDA':12s} {'PBE':12s} {'B3LYP':12s}"
    )
    print("-" * 80)

    for symbol, spin in atoms_list:
        if spin == 0:
            continue  # Тільки відкриті системи

        mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

        expected_s2 = spin * (spin + 2) / 4

        s2_values = [expected_s2]

        # HF
        mf_hf = scf.UHF(mol)
        mf_hf.verbose = 0
        mf_hf.kernel()
        s2_hf = mf_hf.spin_square()[0]
        s2_values.append(s2_hf)

        # DFT
        for xc in ["lda", "pbe", "b3lyp"]:
            mf = dft.UKS(mol)
            mf.xc = xc
            mf.verbose = 0
            mf.kernel()
            s2 = mf.spin_square()[0]
            s2_values.append(s2)

        # Виведення
        row_str = f"{symbol:6s} {spin:4d}"
        for s2 in s2_values:
            row_str += f" {s2:12.6f}"

        # Забруднення
        contamination = s2_hf - expected_s2
        if abs(contamination) > 0.01:
            row_str += "  ← помітне забруднення HF"

        print(row_str)

    print("=" * 80)
    print("\nПримітка: Чисті DFT (LDA, PBE) не мають забруднення")
    print("          Гібридні DFT (B3LYP) мають слабке забруднення")


# Атоми з відкритими оболонками
atoms_open = [
    ("H", 1),
    ("Li", 1),
    ("B", 1),
    ("C", 2),
    ("N", 3),
    ("O", 2),
    ("F", 1),
    ("Na", 1),
]

spin_contamination_analysis(atoms_open)
