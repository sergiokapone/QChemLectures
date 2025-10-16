# Сімейство Pople
mol_631g = gto.M(atom='N 0 0 0', basis='6-31g')
mol_6311g = gto.M(atom='N 0 0 0', basis='6-311g')

# З поляризаційними функціями
mol_631gd = gto.M(atom='N 0 0 0', basis='6-31g*')      # d на важких атомах
mol_631gdp = gto.M(atom='N 0 0 0', basis='6-31g**')    # d на важких, p на H

print(f'6-31g:   {mol_631g.nao_nr()} функцій')
print(f'6-31g*:  {mol_631gd.nao_nr()} функцій')
print(f'6-31g**: {mol_631gdp.nao_nr()} функцій')
