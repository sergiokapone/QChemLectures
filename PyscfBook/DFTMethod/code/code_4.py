mf = dft.UKS(mol)
mf.xc = 'bp86'
energy_bp86 = mf.kernel()
print(f'Енергія C (BP86): {energy_bp86:.8f} Ha')