# ============================================================
# nh3_cam_b3lyp.py
# Розрахунок енергії молекули аміаку з функціоналом CAM-B3LYP
# ============================================================

from pyscf import gto, dft

# Молекула аміаку (NH3)
mol = gto.M(
    atom="N 0 0 0; H 0 0 1.0; H 0.9428 0 -0.333; H -0.9428 0 -0.333",
    basis="def2-tzvp",
    spin=0
)

mf = dft.RKS(mol)
mf.xc = "cam-b3lyp"
mf.verbose = 4

energy_cam = mf.kernel()
print(f"\nЕнергія NH3 (CAM-B3LYP): {energy_cam:.8f} Ha")
