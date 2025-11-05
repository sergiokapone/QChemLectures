# ============================================================
# h2o_b3lyp.py
# DFT-розрахунок з гібридним функціоналом B3LYP
# ============================================================

from pyscf import gto, dft

# Створення молекули води
mol = gto.M(
    atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
    basis="6-31g(d)",
    spin=0
)

# Розрахунок методом B3LYP
mf = dft.RKS(mol)
mf.xc = "b3lyp"
mf.verbose = 4

energy_b3lyp = mf.kernel()
print(f"\nЕнергія H2O (B3LYP): {energy_b3lyp:.8f} Ha")

