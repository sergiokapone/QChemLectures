mf = dft.UKS(mol)
mf.xc = "blyp"  # або 'b88,lyp'
energy_blyp = mf.kernel()
print(f"Енергія C (BLYP): {energy_blyp:.8f} Ha")
