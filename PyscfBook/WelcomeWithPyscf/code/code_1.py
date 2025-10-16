import pyscf
# Вивід версії
print(pyscf.__version__)
# Простий тест
from pyscf import gto, scf
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
mf = scf.RHF(mol)
energy = mf.kernel()
print(f'Енергія H2: {energy:.6f} Ha')