from pyscf import gto, dft

mol = gto.M(atom="Ar 0 0 0", basis="def2-tzvp", spin=0)

# Порівняння M06 функціоналів
m06_functionals = {
    "m06l": "M06-L (0% HF)",
    "m06": "M06 (27% HF)",
    "m062x": "M06-2X (54% HF)",
}

print("Порівняння M06 функціоналів для Ar")
print("=" * 60)

for xc, name in m06_functionals.items():
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.verbose = 0

    try:
        energy = mf.kernel()
        print(f"{name:20s}: {energy:.8f} Ha")
    except:
        print(f"{name:20s}: недоступний")
