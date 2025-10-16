from pyscf import gto, scf
import numpy as np

mol = gto.M(atom='Mn 0 0 0', basis='def2-tzvp', spin=5)
mf = scf.UHF(mol)

# --- Стратегія 1: Level shift
mf.level_shift = 0.3
mf.max_cycle = 100

try:
    energy = mf.kernel()
except:
    print('Не конвергувало з level shift')

# --- Стратегія 2: Newton-Raphson SCF (другий порядок)
if not mf.converged:
    mf = mf.newton()  # Перехід до SCF другого порядку
    energy = mf.kernel()

# --- Стратегія 3: Fractional occupation (часткове заселення)
from pyscf import scf
mf = scf.UHF(mol)
mf = scf.addons.frac_occ(mf)
energy = mf.kernel()

print(f'Фінальна енергія: {energy:.8f} Ha')
print(f'Конвергувало: {mf.converged}')
