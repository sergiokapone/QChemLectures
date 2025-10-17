# ============================================================
# code_35.py - Оптимізація геометрії
# ============================================================
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize

# Початкова геометрія H2 (трохи невірна)
mol = gto.M(
    atom='H 0 0 0; H 0 0 0.8',
    basis='cc-pvtz',
    unit='angstrom'
)

print("Оптимізація геометрії H2 (PBE0/cc-pVTZ)")
print("=" * 60)

# DFT з PBE0
mf = dft.RKS(mol)
mf.xc = 'pbe0'
mf.kernel()

print(f"\nПочаткова відстань: 0.80 Å")
print(f"Енергія до оптимізації: {mf.e_tot:.8f} Hartree")

# Оптимізація
mol_eq = optimize(mf)

print(f"\nОптимізована структура:")
print(mol_eq.atom)
coords = mol_eq.atom_coords()
r_opt = np.linalg.norm(coords[1] - coords[0])
print(f"Рівноважна відстань: {r_opt:.6f} Å")

# Фінальна енергія
mf_opt = dft.RKS(mol_eq)
mf_opt.xc = 'pbe0'
e_opt = mf_opt.kernel()
print(f"Енергія після оптимізації: {e_opt:.8f} Hartree")
print(f"Зниження енергії: {(e_opt - mf.e_tot)*27.211:.6f} eV")

