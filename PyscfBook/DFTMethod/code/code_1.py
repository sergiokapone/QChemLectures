from pyscf import gto, dft

# Приклад LDA розрахунку атома Ne
mol = gto.M(atom="Ne 0 0 0", basis="cc-pvdz", spin=0)

# LDA функціонал (VWN для кореляції)
mf = dft.RKS(mol)
mf.xc = "lda,vwn"  # або просто 'lda'
mf.verbose = 4
energy_lda = mf.kernel()

print(f"\nЕнергія Ne (LDA): {energy_lda:.8f} Ha")

# Порівняння з HF
from pyscf import scf

mf_hf = scf.RHF(mol)
mf_hf.verbose = 0
energy_hf = mf_hf.kernel()

print(f"Енергія Ne (HF):  {energy_hf:.8f} Ha")
print(f"Різниця (LDA-HF): {(energy_lda - energy_hf) * 1000:.2f} mHa")
