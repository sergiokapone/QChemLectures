# ============================================================
# ethanol_coupling_constants.py
# Константи спін-спінової взаємодії для етанолу
# ============================================================

from pyscf import gto, dft
from pyscf.prop.ssc import rhf as ssc_rhf
from pyscf.prop.ssc import rks as ssc_rks

# Етанол CH3CH2OH (одна з конформерів)
mol = gto.M(
    atom="""
    C  -1.1879  -0.3829   0.0000
    C   0.2646   0.0898   0.0000
    O   0.8498  -0.2382   1.2689
    H  -1.7932   0.0427   0.8852
    H  -1.2771  -1.4710   0.0001
    H  -1.7932   0.0427  -0.8852
    H   0.3357   1.1804  -0.0001
    H   0.8239  -0.3263  -0.8524
    H   1.7879   0.0127   1.2891
    """,
    basis="pcJ-1",  # Спеціальний базис для КССВ
    unit="angstrom",
)

print("Константи спін-спінової взаємодії (КССВ) для етанолу")
print("=" * 60)

# DFT розрахунок
print("\nSCF розрахунок (B3LYP/pcJ-1)...")
mf = dft.RKS(mol)
mf.xc = "b3lyp"
mf.verbose = 0
mf.kernel()

# Обчислюємо КССВ
print("\nОбчислення КССВ (це може зайняти хвилину)...")
ssc = ssc_rks.SSC(mf)
j_tot = ssc.kernel()

print("\nПовна таблиця КССВ J (Гц):")
print("-" * 60)

# Виводимо тільки значущі константи (|J| > 1 Гц)
atom_labels = ["C1(CH3)", "C2(CH2)", "O", "H", "H", "H", "H", "H", "H(OH)"]

print(f"\n{'Пара атомів':<25} {'J_tot (Гц)':<12} {'Тип'}")
print("-" * 60)

for i in range(mol.natm):
    for j in range(i+1, mol.natm):
        J_value = j_tot[i, j]

        if abs(J_value) > 1.0:  # Тільки значущі
            sym_i = mol.atom_symbol(i)
            sym_j = mol.atom_symbol(j)

            # Визначаємо тип взаємодії
            bond_distance = np.linalg.norm(
                mol.atom_coord(i) - mol.atom_coord(j)
            ) * 0.529177  # bohr -> Angstrom

            if bond_distance < 1.2:
                j_type = "¹J (geminal)"
            elif bond_distance < 2.5:
                j_type = "²J (geminal)"
            elif bond_distance < 3.5:
                j_type = "³J (vicinal)"
            else:
                j_type = "⁴J (long-range)"

            pair_label = f"{atom_labels[i]} - {atom_labels[j]}"
            print(f"{pair_label:<25} {J_value:>10.2f}  {j_type}")

print("\n" + "=" * 60)
print("Типові експериментальні значення:")
print("  ¹J(CH) для CH₃: ~125 Гц")
print("  ¹J(CH) для CH₂: ~140 Гц")
print("  ³J(HH) vicinal: ~7 Гц (залежить від діедрального кута)")

print("\nПримітка:")
print("- pcJ-n базиси оптимізовані для КССВ")
print("- B3LYP зазвичай дає похибку ±10%")
print("- ³J(HH) залежить від конформації (крива Карплуса)")

