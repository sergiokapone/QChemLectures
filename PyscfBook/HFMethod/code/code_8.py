from pyscf import gto, scf

# Be: [He] 2s², основний стан ¹S
mol = gto.M(
    atom="Be 0 0 0",
    basis="cc-pvtz",
    spin=0,  # замкнена оболонка
    symmetry=True,
)

print("Берилій (Be):")
print("  Конфігурація: [He] 2s²")
print("  Основний стан: ¹S")

# RHF розрахунок
mf = scf.RHF(mol)
energy = mf.kernel()

print(f"\nЕнергія Be: {energy:.8f} Ha")

# Порівняння з експериментом
exp_be = -14.6674  # Ha (експериментальна повна енергія)
print(f"Експериментальна: {exp_be:.4f} Ha")
print(f"Кореляційна енергія: {exp_be - energy:.4f} Ha")
