from pyscf import gto, scf


def ionization_energy(
    symbol, charge_neutral=0, spin_neutral=None, spin_cation=None, basis="cc-pvtz"
):
    """
    Розрахунок енергії іонізації атома
    IE = E(A⁺) - E(A)
    """
    # Нейтральний атом
    mol_neutral = gto.M(
        atom=f"{symbol} 0 0 0",
        basis=basis,
        charge=charge_neutral,
        spin=spin_neutral,
        verbose=0,
    )

    if spin_neutral == 0:
        mf_neutral = scf.RHF(mol_neutral)
    else:
        mf_neutral = scf.UHF(mol_neutral)

    mf_neutral.verbose = 0
    e_neutral = mf_neutral.kernel()

    # Катіон
    mol_cation = gto.M(
        atom=f"{symbol} 0 0 0",
        basis=basis,
        charge=charge_neutral + 1,
        spin=spin_cation,
        verbose=0,
    )

    if spin_cation == 0:
        mf_cation = scf.RHF(mol_cation)
    else:
        mf_cation = scf.UHF(mol_cation)

    mf_cation.verbose = 0
    e_cation = mf_cation.kernel()

    # Енергія іонізації
    ie_ha = e_cation - e_neutral
    ie_ev = ie_ha * 27.211386

    return ie_ev, e_neutral, e_cation


# Розрахунок IE для атомів 2-го періоду
atoms_ie = [
    ("Li", 1, 0),  # Li → Li⁺
    ("Be", 0, 1),  # Be → Be⁺
    ("B", 1, 0),  # B → B⁺
    ("C", 2, 1),  # C → C⁺
    ("N", 3, 2),  # N → N⁺
    ("O", 2, 3),  # O → O⁺
    ("F", 1, 2),  # F → F⁺
    ("Ne", 0, 1),  # Ne → Ne⁺
]

# Експериментальні значення (eV)
exp_ie = {
    "Li": 5.39,
    "Be": 9.32,
    "B": 8.30,
    "C": 11.26,
    "N": 14.53,
    "O": 13.62,
    "F": 17.42,
    "Ne": 21.56,
}

print("Енергії іонізації атомів 2-го періоду")
print("=" * 65)
print(f"{'Атом':4s} {'IE(HF), eV':12s} {'IE(exp), eV':12s} {'Похибка, eV':12s}")
print("-" * 65)

for symbol, spin_n, spin_c in atoms_ie:
    ie_calc, e_n, e_c = ionization_energy(symbol, 0, spin_n, spin_c)
    ie_experimental = exp_ie[symbol]
    error = ie_calc - ie_experimental

    print(f"{symbol:4s} {ie_calc:12.2f} {ie_experimental:12.2f} {error:12.2f}")

print("=" * 65)
