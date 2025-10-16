from pyscf import gto, dft

mol = gto.M(atom="O 0 0 0", basis="aug-cc-pvtz", spin=2)

mf = dft.UKS(mol)
mf.xc = "pbe0"
energy_pbe0 = mf.kernel()

print(f"Енергія O (PBE0): {energy_pbe0:.8f} Ha")
