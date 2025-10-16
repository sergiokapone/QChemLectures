from pyscf import gto, dft

mol = gto.M(atom="Ni 0 0 0", basis="def2-svp", spin=2)

mf = dft.UKS(mol)
mf.xc = "m06l"  # або 'm06-l'
mf.verbose = 4

try:
    energy_m06l = mf.kernel()
    print(f"\nЕнергія Ni (M06-L): {energy_m06l:.8f} Ha")
except:
    print("M06-L може бути недоступний у вашій версії PySCF")
    print("Встановіть: pip install pyscf[geomopt]")
