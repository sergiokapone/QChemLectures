import numpy as np
from pyscf import gto, scf, grad
from pyscf.hessian.rhf import Hessian

# Параметри
h = 1e-4  # крок для чисельних похідних (Å)
bohr2ang = 0.52917721092

# Молекула H2O
mol = gto.M(
    atom = '''
    O  0.0  0.0  0.0
    H  0.0  0.7  0.3
    H  0.0 -0.7 -0.3
    ''',
    basis = 'sto-3g',
    unit = 'Angstrom',
    verbose = 0
)

# Координати з молекули (в Angstrom)
coords = mol.atom_coords(unit='Angstrom')

# SCF розрахунок
mf = scf.RHF(mol).run(verbose=0)
print(f"Енергія: {mf.e_tot:.8f} Hartree\n")

# ==========================================
# АНАЛІТИЧНІ ПОХІДНІ
# ==========================================
print("="*60)
print("АНАЛІТИЧНІ ПОХІДНІ")
print("="*60)

# Градієнт (Hartree/Bohr -> Hartree/Å)
g_anal = grad.RHF(mf).grad().flatten() / bohr2ang
print("Градієнт (Hartree/Å):")
print(g_anal)

# Гессіан (Hartree/Bohr² -> Hartree/Å²)
H_tensor = Hessian(mf).kernel() / (bohr2ang**2)
# Порядок індексів PySCF: [atom1, xyz1, atom2, xyz2]
# Переставляємо: [atom1, xyz1, atom2, xyz2] -> [atom1, atom2, xyz1, xyz2]
H_anal = H_tensor.transpose(0, 2, 1, 3).reshape(9, 9)
print("\nГессіан (Hartree/Å²), форма:", H_anal.shape)
print("Діагональні елементи:", np.diag(H_anal))

# ==========================================
# ЧИСЕЛЬНІ ПОХІДНІ
# ==========================================
print("\n" + "="*60)
print("ЧИСЕЛЬНІ ПОХІДНІ")
print("="*60)

# Градієнт: центральні різниці по енергії
print("Обчислення градієнта...")
g_num = np.zeros(9)
for i in range(9):
    atom, comp = i // 3, i % 3

    # E(x + h)
    coords_p = coords.copy()
    coords_p[atom, comp] += h
    mol_p = gto.M(atom=f"O {coords_p[0,0]} {coords_p[0,1]} {coords_p[0,2]}; "
                       f"H {coords_p[1,0]} {coords_p[1,1]} {coords_p[1,2]}; "
                       f"H {coords_p[2,0]} {coords_p[2,1]} {coords_p[2,2]}",
                  basis='sto-3g', unit='Angstrom', verbose=0)
    E_p = scf.RHF(mol_p).run(verbose=0).e_tot

    # E(x - h)
    coords_m = coords.copy()
    coords_m[atom, comp] -= h
    mol_m = gto.M(atom=f"O {coords_m[0,0]} {coords_m[0,1]} {coords_m[0,2]}; "
                       f"H {coords_m[1,0]} {coords_m[1,1]} {coords_m[1,2]}; "
                       f"H {coords_m[2,0]} {coords_m[2,1]} {coords_m[2,2]}",
                  basis='sto-3g', unit='Angstrom', verbose=0)
    E_m = scf.RHF(mol_m).run(verbose=0).e_tot

    g_num[i] = (E_p - E_m) / (2*h)

print("Градієнт (Hartree/Å):")
print(g_num)

# Гессіан: центральні різниці по градієнтах
print("\nОбчислення гессіана...")
H_num = np.zeros((9,9))
for j in range(9):
    atom, comp = j // 3, j % 3

    # grad(x + h)
    coords_p = coords.copy()
    coords_p[atom, comp] += h
    mol_p = gto.M(atom=f"O {coords_p[0,0]} {coords_p[0,1]} {coords_p[0,2]}; "
                       f"H {coords_p[1,0]} {coords_p[1,1]} {coords_p[1,2]}; "
                       f"H {coords_p[2,0]} {coords_p[2,1]} {coords_p[2,2]}",
                  basis='sto-3g', unit='Angstrom', verbose=0)
    mf_p = scf.RHF(mol_p).run(verbose=0)
    g_p = grad.RHF(mf_p).grad()  # (3, 3) в Hartree/Bohr

    # grad(x - h)
    coords_m = coords.copy()
    coords_m[atom, comp] -= h
    mol_m = gto.M(atom=f"O {coords_m[0,0]} {coords_m[0,1]} {coords_m[0,2]}; "
                       f"H {coords_m[1,0]} {coords_m[1,1]} {coords_m[1,2]}; "
                       f"H {coords_m[2,0]} {coords_m[2,1]} {coords_m[2,2]}",
                  basis='sto-3g', unit='Angstrom', verbose=0)
    mf_m = scf.RHF(mol_m).run(verbose=0)
    g_m = grad.RHF(mf_m).grad()  # (3, 3) в Hartree/Bohr

    # Похідна градієнта: d(grad)/dx
    # grad в Hartree/Bohr, dx в Angstrom -> результат в Hartree/(Bohr*Angstrom)
    # Треба перевести в Hartree/Angstrom²
    dg = (g_p - g_m) / (2*h)  # Hartree/(Bohr*Angstrom)
    dg = dg / bohr2ang  # Hartree/Angstrom²

    H_num[:,j] = dg.flatten()

H_num = 0.5 * (H_num + H_num.T)  # симетризація
print("Гессіан (Hartree/Å²), форма:", H_num.shape)
print("Діагональні елементи:", np.diag(H_num))

# ==========================================
# ПОРІВНЯННЯ
# ==========================================
print("\n" + "="*60)
print("ПОРІВНЯННЯ")
print("="*60)

print("\nГРАДІЄНТ:")
print(f"  Макс. різниця: {np.max(np.abs(g_num - g_anal)):.3e}")
print(f"  Відносна помилка: {np.linalg.norm(g_num - g_anal)/np.linalg.norm(g_anal):.3e}")

print("\nГЕССІАН:")
print(f"  Макс. різниця: {np.max(np.abs(H_num - H_anal)):.3e}")
print(f"  Відносна помилка: {np.linalg.norm(H_num - H_anal)/np.linalg.norm(H_anal):.3e}")

# Порівняння блоків 3x3
print("\nМакс. різниця в блоках гессіана (атом × атом):")
print(f"{'':>6} {'O':>12} {'H1':>12} {'H2':>12}")
for a, name_a in enumerate(['O','H1','H2']):
    print(f"{name_a:>6}", end="")
    for b in range(3):
        block_diff = H_num[3*a:3*a+3, 3*b:3*b+3] - H_anal[3*a:3*a+3, 3*b:3*b+3]
        print(f"{np.max(np.abs(block_diff)):12.3e}", end="")
    print()

