# ============================================================
# code_45.py - Аналіз хвильових функцій
# ============================================================
from pyscf import gto, dft, lo
import numpy as np

print("Детальний аналіз електронної структури CO")
print("=" * 80)

mol = gto.M(
    atom='C 0 0 0; O 0 0 1.128',
    basis='cc-pvtz',
    unit='angstrom',
    symmetry=True
)

mf = dft.RKS(mol)
mf.xc = 'pbe0'
energy = mf.kernel()

print(f"\nПовна енергія: {energy:.8f} Hartree")
print(f"Точкова група: {mol.groupname}")

# Аналіз орбіталей
print("\n" + "=" * 80)
print("Орбітальні енергії (заняті):")
print("-" * 80)

occ_idx = mf.mo_occ > 0
occ_energies = mf.mo_energy[occ_idx]
occ_labels = mol.ao_labels()

print(f"{'#':<5} {'Енергія (eV)':<15} {'Заняття':<10}")
print("-" * 40)
for i, e in enumerate(occ_energies):
    print(f"{i+1:<5} {e*27.211:<15.4f} {mf.mo_occ[i]:<10.1f}")

# HOMO-LUMO gap
homo_idx = np.where(mf.mo_occ > 0)[0][-1]
lumo_idx = np.where(mf.mo_occ == 0)[0][0]
homo_energy = mf.mo_energy[homo_idx]
lumo_energy = mf.mo_energy[lumo_idx]
gap = (lumo_energy - homo_energy) * 27.211

print(f"\nHOMO (орбіталь {homo_idx+1}): {homo_energy*27.211:.4f} eV")
print(f"LUMO (орбіталь {lumo_idx+1}): {lumo_energy*27.211:.4f} eV")
print(f"HOMO-LUMO gap: {gap:.4f} eV")

# Дипольний момент
dipole = mf.dip_moment(unit='Debye')
print(f"\nДипольний момент:")
print(f"  μ_x = {dipole[0]:.4f} D")
print(f"  μ_y = {dipole[1]:.4f} D")
print(f"  μ_z = {dipole[2]:.4f} D")
print(f"  |μ| = {np.linalg.norm(dipole):.4f} D")

# Популяційний аналіз Малікена
print("\nПопуляційний аналіз Малікена:")
pop = mf.mulliken_pop()
print(f"  Заряд на C: {pop[0][0]:.4f}")
print(f"  Заряд на O: {pop[0][1]:.4f}")

# Аналіз зв'язку (bond order)
print("\nПорядок зв'язку (якісна оцінка):")
print("  CO має потрійний зв'язок: σ² π⁴")

