from pyscf import gto, scf
# Атом Літію
mol = gto.M(
atom='Li 0 0 0',
basis='cc-pvdz',
spin=1                # Один неспарений електрон
)
# UHF розрахунок
mf = scf.UHF(mol)
mf.verbose = 4            # Детальний вивід
energy = mf.kernel()
# Аналіз результатів
print('\n=== Результати розрахунку ===')
print(f'Енергія: {energy:.8f} Ha')
print(f'Кількість базисних функцій: {mol.nao_nr()}')
print(f'Кількість електронів: {mol.nelectron}')
print(f'Спін: {mol.spin}')
# Заселеності Малікена
from pyscf import lo
pop = mf.mulliken_pop()
print('\nАналіз заселеностей Малікена:')
print(pop)
# Енергії орбіталей
print('\nЕнергії орбіталей (альфа):')
for i, e in enumerate(mf.mo_energy[0][:5]):
print(f'  MO {i+1}: {e:.6f} Ha ({e*27.211386:.4f} eV)')