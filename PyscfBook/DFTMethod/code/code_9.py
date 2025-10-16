from pyscf import gto, dft

mol = gto.M(atom="N 0 0 0", basis="6-311+g(d,p)", spin=3)

# B3LYP розрахунок
mf = dft.UKS(mol)
mf.xc = "b3lyp"
mf.verbose = 4
energy_b3lyp = mf.kernel()

print(f"\nЕнергія N (B3LYP): {energy_b3lyp:.8f} Ha")

# Аналіз компонентів
print("\nВнесок точного обміну: 20%")
print("Це робить розрахунок повільнішим, але точнішим")
