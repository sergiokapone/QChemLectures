import numpy as np
from pyscf import gto, scf

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

# SCF без поля
mf0 = scf.RHF(mol).run(conv_tol=1e-14, verbose=0)
print(f"Енергія без поля: {mf0.e_tot:.10f} Hartree")
mu0 = mf0.dip_moment(unit='AU', verbose=0)
print(f"Дипольний момент без поля: {mu0}\n")

# ==========================================
# ЧИСЕЛЬНА ПОЛЯРИЗОВАНІСТЬ
# α_ij = -∂μ_i/∂F_j
# ==========================================
print("="*60)
print("ЧИСЕЛЬНИЙ РОЗРАХУНОК ПОЛЯРИЗОВАНОСТІ")
print("="*60)

alpha = np.zeros((3, 3))

# Цикл по компонентах поля (x, y, z)
for j in range(3):
    print(f"\nОбчислення для поля в напрямку {'xyz'[j]}...")

    # --- Поле +h ---
    mol_p = mol.copy()
    mf_p = scf.RHF(mol_p)
    mf_p.verbose = 0
    mf_p.conv_tol = 1e-14

    # Модифікуємо гамільтоніан: H_eff = H₀ - F·r
    get_hcore_orig = mf_p.get_hcore
    field_p = np.zeros(3)
    field_p[j] = h

    with mol_p.with_common_orig((0,0,0)):
        dip_ints = mol_p.intor_symmetric('int1e_r', comp=3)

    mf_p.get_hcore = lambda *args: get_hcore_orig() - np.einsum('x,xij->ij', field_p, dip_ints)
    mf_p.kernel()
    mu_p = mf_p.dip_moment(unit='AU', verbose=0)

    # --- Поле -h ---
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

    # Центральна різниця: α[:,j] = -(μ(+F) - μ(-F))/(2h)
    alpha[:, j] = -(mu_p - mu_m) / (2*h)

    print(f"  μ(+F_{j}): {mu_p}")
    print(f"  μ(-F_{j}): {mu_m}")
    print(f"  dμ/dF_{j}: {-(mu_p - mu_m)/(2*h)}")

# Результат
print("\n" + "="*60)
print("ТЕНЗОР ПОЛЯРИЗОВАНОСТІ α (a.u., 3×3)")
print("="*60)
print(alpha)
print("\nДіагональні компоненти (α_xx, α_yy, α_zz):")
print(np.diag(alpha))
print("\nСередня поляризованість ⟨α⟩ = Tr(α)/3:")
print(np.trace(alpha) / 3)

