from pyscf import gto, dft
import os

# Налаштування кількості потоків
os.environ["OMP_NUM_THREADS"] = "4"  # 4 потоки

mol = gto.M(atom="Br 0 0 0", basis="def2-tzvp", spin=1, verbose=4)

mf = dft.UKS(mol)
mf.xc = "pbe0"

# DFT розрахунки автоматично використовують паралелізацію
# для обчислення інтегралів та Фок-матриць
energy = mf.kernel()

print(f"\nЕнергія Br: {energy:.8f} Ha")
print(f"Використано потоків: {os.environ.get('OMP_NUM_THREADS')}")
