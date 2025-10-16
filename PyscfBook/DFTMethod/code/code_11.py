from pyscf import gto, dft

mol = gto.M(atom="F 0 0 0", basis="cc-pvqz", spin=1)

mf = dft.UKS(mol)
mf.xc = "camb3lyp"  # або 'cam-b3lyp'
energy_camb3lyp = mf.kernel()

print(f"Енергія F (CAM-B3LYP): {energy_camb3lyp:.8f} Ha")
