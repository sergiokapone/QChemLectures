from pyscf import gto

# Простий спосіб
mol = gto.Mole(
    atom='Li 0 0 0',
    basis='6-31g',
    charge=0,
    spin=1
)
