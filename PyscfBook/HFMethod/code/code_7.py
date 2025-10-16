from pyscf import gto, scf

# Li: [He] 2s¹, основний стан ²S
mol = gto.M(
    atom="Li 0 0 0",
    basis="cc-pvdz",
    spin=1,  # один неспарений електрон (S=1/2)
    symmetry=True,
)

print("Літій (Li):")
print("  Конфігурація: [He] 2s¹")
print("  Основний стан: ²S")
print(f"  Електронів: {mol.nelectron}")

# UHF розрахунок
mf = scf.UHF(mol)
energy = mf.kernel()

print(f"\nЕнергія Li: {energy:.8f} Ha")
print(f"Енергія Li: {energy * 27.211386:.4f} eV")

# Аналіз заселеностей
print("\nЗаселеності Малікена:")
pop = mf.mulliken_pop()

# Орбітальний аналіз
print("\nАльфа-орбіталі (заповнені):")
for i in range(mol.nelec[0]):
    print(f"  α-MO {i + 1}: {mf.mo_energy[0][i]:10.6f} Ha")

print("\nБета-орбіталі (заповнені):")
for i in range(mol.nelec[1]):
    print(f"  β-MO {i + 1}: {mf.mo_energy[1][i]:10.6f} Ha")

# Спінова густина
s2 = mf.spin_square()
print(f"\n<S²> = {s2[0]:.6f} (очікується 0.75)")
