from pyscf import gto, scf

# Атом He (основний стан ¹S)
mol = gto.M(
    atom="He 0 0 0",
    basis="cc-pvtz",
    spin=0,  # Замкнена оболонка
    verbose=4,
)

print("Електронна конфігурація: 1s²")
print(f"Кількість електронів: {mol.nelectron}")

# RHF для замкненої оболонки
mf = scf.RHF(mol)
energy = mf.kernel()

print(f"\nЕнергія He (HF): {energy:.8f} Ha")
print(f"Енергія He (HF): {energy * 27.211386:.6f} eV")

# Експериментальна енергія He: -2.90372 Ha
exp_energy = -2.90372
correlation_energy = exp_energy - energy

print(f"\nЕкспериментальна енергія: {exp_energy:.6f} Ha")
print(f"Кореляційна енергія: {correlation_energy:.6f} Ha")
print(f"Відносна похибка HF: {abs(correlation_energy / exp_energy) * 100:.2f}%")
