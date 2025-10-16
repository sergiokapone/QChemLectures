from pyscf import gto, scf
import numpy as np

# Створення атома Гідрогену
mol = gto.M(
    atom="H 0 0 0",  # координати ядра
    basis="sto-3g",  # мінімальний базис
    spin=1,  # один неспарений електрон
    verbose=4,  # рівень деталізації виводу
)

print(f"Кількість електронів: {mol.nelectron}")
print(f"Базисних функцій: {mol.nao_nr()}")

# Розрахунок методом UHF (необмежений Гартрі-Фок)
mf = scf.UHF(mol)
energy = mf.kernel()

print(f"\nЕнергія H (STO-3G): {energy:.8f} Ha")
print(f"Енергія H (STO-3G): {energy * 27.211386:.6f} eV")

# Теоретичне значення
print(f"Теоретична енергія: -0.5 Ha")
print(f"Похибка базису: {abs(energy + 0.5):.6f} Ha")
