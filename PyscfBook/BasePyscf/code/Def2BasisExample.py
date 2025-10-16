# Сімейство Ahlrichs
mol_svp = gto.M(atom='Si 0 0 0', basis='def2-svp')
mol_tzvp = gto.M(atom='Si 0 0 0', basis='def2-tzvp')
mol_qzvp = gto.M(atom='Si 0 0 0', basis='def2-qzvp')

print(f'def2-SVP:  {mol_svp.nao_nr()} функцій')
print(f'def2-TZVP: {mol_tzvp.nao_nr()} функцій')
print(f'def2-QZVP: {mol_qzvp.nao_nr()} функцій')
