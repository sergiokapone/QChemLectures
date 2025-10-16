# cc-pVXZ сімейство (X = D, T, Q, 5, 6)
mol_dz = gto.M(atom='O 0 0 0', basis='cc-pvdz')   # Double-zeta
mol_tz = gto.M(atom='O 0 0 0', basis='cc-pvtz')   # Triple-zeta
mol_qz = gto.M(atom='O 0 0 0', basis='cc-pvqz')   # Quadruple-zeta

print(f'cc-pVDZ: {mol_dz.nao_nr()} функцій')
print(f'cc-pVTZ: {mol_tz.nao_nr()} функцій')
print(f'cc-pVQZ: {mol_qz.nao_nr()} функцій')

# Augmented версії (з дифузними функціями)
mol_adz = gto.M(atom='O 0 0 0', basis='aug-cc-pvdz')
print(f'aug-cc-pVDZ: {mol_adz.nao_nr()} функцій')
