import numpy as np
from pyscf import gto

mol = gto.M(atom='Si 0 0 0', basis='6-31g*')

# Отримання базисних функцій
print('Базисні функції (AO labels):')
for i, label in enumerate(mol.ao_labels(fmt=False)):
    atom_id, atom_symbol, shell_type, *rest = label
    print(f'{i:3d}: Атом {atom_symbol}, тип {shell_type}')

# Кількість функцій кожного типу
from collections import Counter
ao_types = [label[2] for label in mol.ao_labels(fmt=False)]
count = Counter(ao_types)

print(f'\nРозподіл по типах:')
for shell_type, num in sorted(count.items()):
    print(f'  {shell_type}: {num} функцій')

# Перекривання базисних функцій
s = mol.intor('int1e_ovlp')
print(f'\nМатриця перекривання: {s.shape}')
print(f'Діагональні елементи (норми): {np.diag(s)[:5]}')

# Загальна кількість AO
print(f'\nЗагальна кількість AO: {mol.nao_nr()}')
