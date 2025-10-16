from pyscf import gto, dft

mol = gto.M(atom="C 0 0 0", basis="def2-qzvp", spin=2)

mf = dft.UKS(mol)
mf.xc = "scan"
energy_scan = mf.kernel()

print(f"Енергія C (SCAN): {energy_scan:.8f} Ha")
