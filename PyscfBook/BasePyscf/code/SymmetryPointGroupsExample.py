from pyscf import gto

# Автоматичне визначення симетрії
mol = gto.M(
    atom='Ne 0 0 0',
    basis='cc-pvdz',
    symmetry=True  # Автоматичне визначення точкової групи
)

print(f'Точкова група: {mol.groupname}')
print(f'Топологічна група: {mol.topgroup}')

# Для атома результат, як правило: D2h або подібна підгрупа
