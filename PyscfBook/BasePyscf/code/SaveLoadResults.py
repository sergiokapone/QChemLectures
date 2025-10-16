from pyscf import gto, scf, lib

mol = gto.M(atom='Cu 0 0 0', basis='def2-tzvp', spin=1)
mf = scf.UHF(mol)

# Збереження у checkpoint файл
mf.chkfile = 'cu_atom.chk'
energy = mf.kernel()

# Завантаження результатів з файлу
mol2 = lib.chkfile.load_mol('cu_atom.chk')
mf2 = scf.UHF(mol2)
mf2.__dict__.update(lib.chkfile.load('cu_atom.chk', 'scf'))

print('Завантажена енергія:', mf2.e_tot)

# Використання як початкового наближення для нового базису
mol3 = gto.M(atom='Cu 0 0 0', basis='def2-qzvp', spin=1)
mf3 = scf.UHF(mol3)
mf3.init_guess = 'chkfile'
mf3.chkfile = 'cu_atom.chk'
energy3 = mf3.kernel()
