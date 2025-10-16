from pyscf import gto, scf

# --- 1. Визначення молекули ---
mol = gto.M(
    atom="Li 0 0 0",
    basis="cc-pvdz",
    spin=1,          # неспарений електрон (S=1/2)
    symmetry=True,
)

print("Літій (Li):")
print("  Конфігурація: [He] 2s¹")
print("  Основний стан: ²S")
print(f"  Електронів: {mol.nelectron}")
print()

# --- 2. UHF ---
uhf = scf.UHF(mol)
E_uhf = uhf.kernel()

print("=== UHF ===")
print(f"Енергія Li (UHF): {E_uhf:.8f} Ha")
print(f"Енергія Li (UHF): {E_uhf * 27.211386:.4f} eV")

s2_uhf = uhf.spin_square()
print(f"<S²> = {s2_uhf[0]:.6f}  (очікується 0.75)\n")

# --- 3. ROHF ---
rohf = scf.ROHF(mol)
E_rohf = rohf.kernel()

print("=== ROHF ===")
print(f"Енергія Li (ROHF): {E_rohf:.8f} Ha")
print(f"Енергія Li (ROHF): {E_rohf * 27.211386:.4f} eV")

s2_rohf = rohf.spin_square()
print(f"<S²> = {s2_rohf[0]:.6f}  (очікується 0.75)\n")

# --- 4. Порівняння ---
ΔE = (E_uhf - E_rohf) * 27.211386
print(f"Різниця (UHF − ROHF): {ΔE:.6f} eV")

# --- 5. Коментар ---
# UHF дозволяє різні орбіталі для α і β спінів,
# тому виникає спінове забруднення (<S²> > 0.75).
# ROHF зберігає правильну спінову симетрію, але менш гнучкий варіаційно.

