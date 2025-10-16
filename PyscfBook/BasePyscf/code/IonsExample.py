from pyscf import gto, scf

# Нейтральний атом Li
li_atom = gto.M(atom='Li 0 0 0', basis='6-31g', charge=0, spin=1)

# Катіон Li+ (втрачено 1 електрон)
li_cation = gto.M(atom='Li 0 0 0', basis='6-31g', charge=1, spin=0)

# Аніон Li- (додано 1 електрон)
li_anion = gto.M(atom='Li 0 0 0', basis='6-31g', charge=-1, spin=1)

print(f'Li:  {li_atom.nelectron} електронів')
print(f'Li+: {li_cation.nelectron} електронів')
print(f'Li-: {li_anion.nelectron} електронів')
