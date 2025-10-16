from pyscf import gto, scf
# Атом Гелію
mol = gto.M(
atom='He 0 0 0',
basis='6-31g',
spin=0,               # Всі електрони спарені
charge=0              # Нейтральний атом
)
# RHF для замкненої оболонки
mf = scf.RHF(mol)
energy = mf.kernel()
print(f'Енергія He: {energy:.8f} Ha')