from pyscf import gto, dft

mol = gto.M(atom="C 0 0 0", basis="cc-pvtz", spin=2)
mf = dft.UKS(mol)
mf.xc = "pbe"  # або 'pbe,pbe'
energy_pbe = mf.kernel()
print(f"Енергія C (PBE): {energy_pbe:.8f} Ha")
