# Автоматичне визначення симетрії
mol = gto.M(atom='Ne 0 0 0', symmetry=True)
print(f'Точкова група: {mol.groupname}')
# Вимкнення симетрії
mol = gto.M(atom='Ne 0 0 0', symmetry=False)