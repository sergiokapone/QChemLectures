from pyscf import gto
from pyscf.data import elements

mol = gto.M(atom='Zn 0 0 0', basis='def2-svp')

# Основні атомні характеристики
symbol = mol.atom_symbol(0)
charge = mol.atom_charge(0)
z = gto.charge(symbol)

print(f'Атом: {symbol}')
print(f'Атомний номер: {z}')
print(f'Заряд ядра: {charge}')
print(f'Атомна маса: {elements.MASSES[charge]:.4f} а.о.м.')
print(f'Ковалентний радіус: {elements.COV_RADII[charge]:.2f} Å')

# Кількість електронів (нейтральний атом: ne = Z)
nelectron = z - mol.charge
print(f'Електронів: {nelectron}')
