"""
scan_vs_pbe_h2o.py
Порівняння meta-GGA функціонала SCAN з PBE на прикладі молекули H2O.
SCAN краще описує водневі зв'язки та локалізовану густину,
забезпечуючи більш фізично точну повну енергію.
"""

from pyscf import gto, dft

# Геометрія молекули води (в ангстремах)
mol = gto.M(
    atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
    basis="def2-tzvp",
    spin=0
)

# ====== SCAN ======
mf_scan = dft.RKS(mol)
mf_scan.xc = "scan"
energy_scan = mf_scan.kernel()
print(f"Енергія H2O (SCAN): {energy_scan:.8f} Ha")

# ====== PBE ======
mf_pbe = dft.RKS(mol)
mf_pbe.xc = "pbe"
energy_pbe = mf_pbe.kernel()
print(f"Енергія H2O (PBE):  {energy_pbe:.8f} Ha")

# Порівняння
print(f"Різниця (SCAN - PBE): {(energy_scan - energy_pbe) * 1000:.2f} mHa")
