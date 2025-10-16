from pyscf import gto

mol = gto.M(atom='O 0 0 0', basis='6-31g', spin=2)

# Інформація про систему
print(f'Кількість електронів: {mol.nelectron}')
print(f'Заряд: {mol.charge}')
print(f'Спін (2S): {mol.spin}')
print(f'Кількість базисних функцій: {mol.nao_nr()}')

# Альфа та бета електрони
print(f'Альфа електронів: {mol.nelec[0]}')
print(f'Бета електронів: {mol.nelec[1]}')

# Атомна структура
print(f'Кількість атомів: {mol.natm}')
print(f'Заряди ядер: {mol.atom_charges()}')

# Базисний набір
print(f'Назва базису: {mol.basis}')
