import numpy as np
from pyscf import gto, scf
from pyscf.prop.polarizability.rhf import Polarizability

# Параметри
h = 1e-4  # крок для електричного поля (a.u.)

# Молекула H2O
mol = gto.M(
    atom = '''
    O  0.000000  0.000000  0.000000
    H  0.000000 -0.757000  0.587000
    H  0.000000  0.757000  0.587000
    ''',
    basis = '6-31g',
    verbose = 0
)

# SCF розрахунок без поля
mf0 = scf.RHF(mol).run(conv_tol=1e-14, verbose=0)
print(f"Енергія без поля: {mf0.e_tot:.10f} Hartree")
mu0 = mf0.dip_moment(unit='AU', verbose=0)
print(f"Дипольний момент без поля: {mu0}\n")

# ==========================================
# АНАЛІТИЧНА ПОЛЯРИЗОВАНІСТЬ
# ==========================================
print("="*60)
print("АНАЛІТИЧНА ПОЛЯРИЗОВАНІСТЬ")
print("="*60)

pol = Polarizability(mf0)
alpha_anal = pol.polarizability()
print("Тензор поляризованості α (a.u., 3×3):")
print(alpha_anal)
print("\nДіагональні компоненти:", np.diag(alpha_anal))

# ==========================================
# ЧИСЕЛЬНА ПОЛЯРИЗОВАНІСТЬ
# ==========================================
print("\n" + "="*60)
print("ЧИСЕЛЬНА ПОЛЯРИЗОВАНІСТЬ")
print("="*60)

# Метод 1: через дипольний момент
# α_ij = -∂μ_i/∂F_j ≈ -(μ_i(+F_j) - μ_i(-F_j))/(2h)
print("Обчислення через дипольний момент (µ)...")

alpha_num = np.zeros((3, 3))

for j in range(3):  # компонента поля
    # Додаємо поле через модифікацію одноелектронного гамільтоніана
    # E = E0 - μ·F, тому додаємо -F·r до h_core

    # Поле +h в напрямку j
    mol_p = mol.copy()
    mf_p = scf.RHF(mol_p)
    mf_p.verbose = 0
    mf_p.conv_tol = 1e-14
    # Зберігаємо оригінальний get_hcore
    get_hcore_orig = mf_p.get_hcore
    # Додаємо електричне поле через дипольні інтеграли
    # H_eff = H₀ - F·r (взаємодія диполя з полем)
    field_p = np.zeros(3)
    field_p[j] = h
    with mol_p.with_common_orig((0,0,0)):
        dip_ints = mol_p.intor_symmetric('int1e_r', comp=3)  # 3 компоненти r
    mf_p.get_hcore = lambda *args: get_hcore_orig() - np.einsum('x,xij->ij', field_p, dip_ints)
    mf_p.kernel()
    mu_p = mf_p.dip_moment(unit='AU', verbose=0)

    # Поле -h в напрямку j
    mol_m = mol.copy()
    mf_m = scf.RHF(mol_m)
    mf_m.verbose = 0
    mf_m.conv_tol = 1e-14
    field_m = np.zeros(3)
    field_m[j] = -h
    with mol_m.with_common_orig((0,0,0)):
        dip_ints = mol_m.intor_symmetric('int1e_r', comp=3)
    mf_m.get_hcore = lambda *args: get_hcore_orig() - np.einsum('x,xij->ij', field_m, dip_ints)
    mf_m.kernel()
    mu_m = mf_m.dip_moment(unit='AU', verbose=0)

    # α[:,j] = -(dμ/dF)
    alpha_num[:, j] = -(mu_p - mu_m) / (2*h)

print("Тензор поляризованості α (a.u., 3×3):")
print(alpha_num)
print("\nДіагональні компоненти:", np.diag(alpha_num))

# ==========================================
# ПОРІВНЯННЯ
# ==========================================
print("\n" + "="*60)
print("ПОРІВНЯННЯ")
print("="*60)

diff = alpha_num - alpha_anal
print("\nРізниця (чисельна - аналітична):")
print(diff)
print(f"\nМакс. абсолютна різниця: {np.max(np.abs(diff)):.3e}")
print(f"Відносна помилка (Frobenius): {np.linalg.norm(diff)/np.linalg.norm(alpha_anal):.3e}")

# Порівняння компонент
print("\nПорівняння компонент:")
labels = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
print(f"{'Комп':<6} {'Чисельна':>12} {'Аналітична':>12} {'Різниця':>12}")
print("-"*48)
for i in range(3):
    for j in range(3):
        idx = i*3 + j
        an = alpha_anal[i,j]
        num = alpha_num[i,j]
        d = num - an
        print(f"{labels[idx]:<6} {num:12.6f} {an:12.6f} {d:12.3e}")

