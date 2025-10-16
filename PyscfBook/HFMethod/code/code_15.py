from pyscf import gto, scf

# Атом C: [He] 2s² 2p², основний стан ³P
basis = "cc-pvtz"

print("Порівняння методів для атома Вуглецю (³P)")
print("=" * 70)

# Молекула
mol = gto.M(
    atom="C 0 0 0",
    basis=basis,
    spin=2,  # Триплет
    symmetry=True,
    verbose=0,
)

# UHF
print("\n1. Unrestricted Hartree-Fock (UHF)")
print("-" * 70)
mf_uhf = scf.UHF(mol)
mf_uhf.verbose = 4
e_uhf = mf_uhf.kernel()

s2_uhf = mf_uhf.spin_square()
print(f"\nЕнергія (UHF): {e_uhf:.8f} Ha")
print(f"<S²> (UHF): {s2_uhf[0]:.6f} (очікується 2.0)")
print(f"Забруднення спіном: {s2_uhf[0] - 2.0:.6f}")

# ROHF
print("\n2. Restricted Open-shell Hartree-Fock (ROHF)")
print("-" * 70)
mf_rohf = scf.ROHF(mol)
mf_rohf.verbose = 4
e_rohf = mf_rohf.kernel()

s2_rohf = mf_rohf.spin_square()
print(f"\nЕнергія (ROHF): {e_rohf:.8f} Ha")
print(f"<S²> (ROHF): {s2_rohf[0]:.6f} (очікується 2.0)")

# Порівняння
print("\n" + "=" * 70)
print("ПОРІВНЯННЯ")
print("=" * 70)
print(f"ΔE (UHF-ROHF): {(e_uhf - e_rohf) * 1000:.4f} mHa")
print(f"ΔE (UHF-ROHF): {(e_uhf - e_rohf) * 627.509:.4f} kcal/mol")

print("\nЗабруднення спіном:")
print(f"  UHF:  {s2_uhf[0] - 2.0:.6f}")
print(f"  ROHF: {s2_rohf[0] - 2.0:.6f}")

# Орбітальні енергії
print("\nОрбітальні енергії (Ha):")
print(f"{'Орбіталь':15s} {'UHF (α)':12s} {'UHF (β)':12s} {'ROHF':12s}")
print("-" * 70)

n_show = 5
for i in range(n_show):
    label = mol.ao_labels()[i] if i < len(mol.ao_labels()) else f"MO{i + 1}"
    e_uhf_a = mf_uhf.mo_energy[0][i]
    e_uhf_b = mf_uhf.mo_energy[1][i]
    e_rohf_i = mf_rohf.mo_energy[i]

    print(f"{label:15s} {e_uhf_a:12.6f} {e_uhf_b:12.6f} {e_rohf_i:12.6f}")
