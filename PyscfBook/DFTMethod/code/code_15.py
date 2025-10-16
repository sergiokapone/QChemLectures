from pyscf import gto, dft

# Замкнена оболонка (RKS)
mol_he = gto.M(atom="He 0 0 0", basis="cc-pvtz", spin=0)
mf_he = dft.RKS(mol_he)
mf_he.xc = "pbe0"
e_he = mf_he.kernel()

print(f"He (RKS/PBE0): {e_he:.8f} Ha")

# Відкрита оболонка (UKS)
mol_li = gto.M(atom="Li 0 0 0", basis="cc-pvtz", spin=1)
mf_li = dft.UKS(mol_li)
mf_li.xc = "pbe0"
e_li = mf_li.kernel()

print(f"Li (UKS/PBE0): {e_li:.8f} Ha")
