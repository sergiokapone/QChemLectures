from pyscf import gto, scf

mol = gto.M(atom="He 0 0 0", basis="6-31g", spin=0)

# RHF розрахунок
mf_rhf = scf.RHF(mol)
mf_rhf.verbose = 0
e_rhf = mf_rhf.kernel()

# UHF розрахунок
mf_uhf = scf.UHF(mol)
mf_uhf.verbose = 0
e_uhf = mf_uhf.kernel()

print(f"RHF енергія: {e_rhf:.10f} Ha")
print(f"UHF енергія: {e_uhf:.10f} Ha")
print(f"Різниця: {abs(e_rhf - e_uhf):.2e} Ha")

# Перевірка симетрії спіну
s2_uhf = mf_uhf.spin_square()
print(f"\n<S²> (UHF): {s2_uhf[0]:.6f}")
print("<S²> (точно): 0.000000")
