from pyscf import gto, scf

mol = gto.M(atom='Ne 0 0 0', basis='cc-pvtz', symmetry=True)
mf = scf.RHF(mol)
energy = mf.kernel()

# Детальні компоненти енергії
print('Енергетичні компоненти:')
print(f'  Електрон-ядерна: {mf.energy_nuc():.8f} Ha')

# Енергетичні вкладення через методи mf
e_elec, e_coul_xc = mf.energy_elec()
print(f'  Електронна (E_elec): {e_elec:.8f} Ha')
print(f'  Кулон+обмін/кореляція: {e_coul_xc:.8f} Ha')  # для DFT це Coulomb+XC

print(f'  Повна енергія: {energy:.8f} Ha')

# Матриці Фока та густини
h1e = mf.get_hcore()      # Одноелектронний гамільтоніан
dm = mf.make_rdm1()       # Матриця густини
vhf = mf.get_veff()       # Хартрі–Фок / ефективний потенціал

print(f'\nРозміри матриць:')
print(f'  h1e: {h1e.shape}')
print(f'  dm: {dm.shape}')
print(f'  vhf: {vhf.shape}')

# Орбітальні енергії
mo_energy = mf.mo_energy
print(f'\nЕнергії МО (перші 5): {mo_energy[:5]}')
