from pyscf import gto, scf
import numpy as np

mol = gto.M(atom="H 0 0 0", basis="cc-pvtz", spin=1)
mf = scf.UHF(mol)
energy = mf.kernel()

# Орбітальні енергії
print("\nОрбітальні енергії (альфа-спін):")
print("-" * 50)
for i, (e, label) in enumerate(zip(mf.mo_energy[0], mol.ao_labels())):
    occ = "(occ)" if i < mol.nelec[0] else "(virt)"
    print(f"{i + 1:2d}. {label:20s}: {e:10.6f} Ha {occ}")

print(f"\nЕнергія 1s орбіталі: {mf.mo_energy[0][0]:.8f} Ha")
print(f"Теоретична енергія:  -0.50000000 Ha")

# Матриця густини
dm = mf.make_rdm1()
print(f"\nМатриця густини (альфа): {dm[0].shape}")
print(f"След dm: {np.trace(dm[0]):.1f} (має дорівнювати 1)")
