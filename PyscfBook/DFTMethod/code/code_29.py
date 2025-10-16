from pyscf import gto, scf, dft
import numpy as np

# Експериментальні дані (eV)
experimental_data = {
    "Li": {"IE1": 5.39, "EA": 0.62},
    "C": {"IE1": 11.26, "EA": 1.26},
    "N": {"IE1": 14.53, "EA": -0.07},
    "O": {"IE1": 13.62, "EA": 1.46},
    "F": {"IE1": 17.42, "EA": 3.40},
    "Ne": {"IE1": 21.56, "EA": None},
}


def benchmark_functional(functional, basis="aug-cc-pvtz"):
    """
    Тестування функціоналу на наборі атомів
    """

    print(f"\nТестування функціоналу {functional.upper()}")
    print("=" * 80)

    mae_ie = []
    mae_ea = []

    for symbol, data in experimental_data.items():
        # Визначення спінів (спрощено)
        spins = {
            "Li": (1, 0),
            "C": (2, 1),
            "N": (3, 2),
            "O": (2, 3),
            "F": (1, 2),
            "Ne": (0, 1),
        }
        spin_n, spin_c = spins[symbol]

        # Нейтральний атом
        mol_n = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin_n, verbose=0)

        if functional.lower() == "hf":
            mf_n = scf.UHF(mol_n) if spin_n > 0 else scf.RHF(mol_n)
        else:
            mf_n = dft.UKS(mol_n) if spin_n > 0 else dft.RKS(mol_n)
            mf_n.xc = functional

        mf_n.verbose = 0
        e_n = mf_n.kernel()

        # Катіон (IE)
        mol_c = gto.M(
            atom=f"{symbol} 0 0 0", basis=basis, charge=1, spin=spin_c, verbose=0
        )

        if functional.lower() == "hf":
            mf_c = scf.UHF(mol_c) if spin_c > 0 else scf.RHF(mol_c)
        else:
            mf_c = dft.UKS(mol_c) if spin_c > 0 else dft.RKS(mol_c)
            mf_c.xc = functional

        mf_c.verbose = 0
        e_c = mf_c.kernel()

        ie_calc = (e_c - e_n) * 27.211386
        ie_exp = data["IE1"]
        error_ie = abs(ie_calc - ie_exp)
        mae_ie.append(error_ie)

        print(
            f"{symbol:4s} IE: calc={ie_calc:7.3f} eV, "
            f"exp={ie_exp:7.3f} eV, error={error_ie:6.3f} eV"
        )

        # Аніон (EA) - якщо є дані
        if data["EA"] is not None and data["EA"] > 0:
            spin_a = spin_n + 1  # Спрощене припущення

            mol_a = gto.M(
                atom=f"{symbol} 0 0 0", basis=basis, charge=-1, spin=spin_a, verbose=0
            )

            if functional.lower() == "hf":
                mf_a = scf.UHF(mol_a) if spin_a > 0 else scf.RHF(mol_a)
            else:
                mf_a = dft.UKS(mol_a) if spin_a > 0 else dft.RKS(mol_a)
                mf_a.xc = functional

            mf_a.verbose = 0
            mf_a.level_shift = 0.5

            try:
                e_a = mf_a.kernel()
                ea_calc = (e_n - e_a) * 27.211386
                ea_exp = data["EA"]
                error_ea = abs(ea_calc - ea_exp)
                mae_ea.append(error_ea)

                print(
                    f"     EA: calc={ea_calc:7.3f} eV, "
                    f"exp={ea_exp:7.3f} eV, error={error_ea:6.3f} eV"
                )
            except:
                print(f"     EA: не конвергувало")

    print("=" * 80)
    print(f"MAE (IE): {np.mean(mae_ie):.3f} eV")
    if mae_ea:
        print(f"MAE (EA): {np.mean(mae_ea):.3f} eV")

    return np.mean(mae_ie), np.mean(mae_ea) if mae_ea else None


# Тестування різних функціоналів
functionals_to_test = ["HF", "LDA", "PBE", "BLYP", "B3LYP", "PBE0"]

results = {}
for func in functionals_to_test:
    mae_ie, mae_ea = benchmark_functional(func)
    results[func] = {"IE": mae_ie, "EA": mae_ea}

# Підсумкова таблиця
print("\n\nПідсумкові MAE (eV):")
print("=" * 50)
print(f"{'Функціонал':12s} {'IE':10s} {'EA':10s}")
print("-" * 50)
for func, res in results.items():
    ea_str = f"{res['EA']:.3f}" if res["EA"] else "N/A"
    print(f"{func:12s} {res['IE']:10.3f} {ea_str:10s}")
