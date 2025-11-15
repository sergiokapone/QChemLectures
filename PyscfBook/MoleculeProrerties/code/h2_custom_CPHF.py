import numpy as np
from pyscf import gto, scf

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

# SCF розрахунок
mf = scf.RHF(mol).run(conv_tol=1e-12, verbose=0)
print(f"Енергія: {mf.e_tot:.10f} Hartree\n")

# Отримуємо дані з SCF
mo_coeff = mf.mo_coeff  # МО коефіцієнти
mo_energy = mf.mo_energy  # МО енергії
mo_occ = mf.mo_occ  # окупації
nocc = np.sum(mo_occ > 0).astype(int)  # кількість зайнятих орбіталей
nvir = len(mo_energy) - nocc  # кількість віртуальних орбіталей

print("="*60)
print("CPHF РОЗРАХУНОК ПОЛЯРИЗОВАНОСТІ")
print("="*60)
print(f"Зайнятих МО: {nocc}")
print(f"Віртуальних МО: {nvir}")
print(f"Розмір задачі: {nocc * nvir} коефіцієнтів на компоненту поля\n")

# Дипольні інтеграли в АО базисі
with mol.with_common_orig((0,0,0)):
    dip_ints_ao = mol.intor_symmetric('int1e_r', comp=3)  # (3, nao, nao)

# Перетворюємо в МО базис: μ^MO = C^T · μ^AO · C
dip_ints_mo = np.einsum('pi,xpq,qj->xij', mo_coeff, dip_ints_ao, mo_coeff)

# Двоелектронні інтеграли (ERI) в МО базисі
# Для CPHF потрібні тільки (ia|jb) блоки
print("Обчислення двоелектронних інтегралів...")
eri_ao = mol.intor('int2e')  # (nao, nao, nao, nao)
# Перетворення в МО базис: потрібні (ia|jb), (ij|ab), (ib|ja)
# i,j - occupied, a,b - virtual
C_occ = mo_coeff[:, :nocc]
C_vir = mo_coeff[:, nocc:]

eri_iajb = np.einsum('pi,pqrs->iqrs', C_occ, eri_ao)
eri_iajb = np.einsum('qa,iqrs->iars', C_vir, eri_iajb)
eri_iajb = np.einsum('rj,iars->iajs', C_occ, eri_iajb)
eri_iajb = np.einsum('sb,iajs->iajb', C_vir , eri_iajb)
# Форма: (nocc, nvir, nocc, nvir) = (i, a, j, b)

eri_ijab = np.einsum('pi,pqrs->iqrs', C_occ, eri_ao)
eri_ijab = np.einsum('qj,iqrs->ijrs', C_occ, eri_ijab)
eri_ijab = np.einsum('ra,ijrs->ijas', C_vir, eri_ijab)
eri_ijab = np.einsum('sb,ijas->ijab', C_vir, eri_ijab)
# Форма: (nocc, nocc, nvir, nvir) = (i, j, a, b)

eri_ibja = np.einsum('pi,pqrs->iqrs', C_occ, eri_ao)
eri_ibja = np.einsum('qb,iqrs->ibrs', C_vir, eri_ibja)
eri_ibja = np.einsum('rj,ibrs->ibjs', C_occ, eri_ibja)
eri_ibja = np.einsum('sa,ibjs->ibja', C_vir, eri_ibja)
# Форма: (nocc, nvir, nocc, nvir) = (i, b, j, a)

print("Побудова матриці H (електронний гесіан)...")
# Матриця H для CPHF рівнянь: H · U = -g
# H[ia,jb] = δ_ij δ_ab (ε_a - ε_i) + 4(ia|jb) - (ij|ab) - (ib|ja)
# Розмірність: (nocc*nvir, nocc*nvir)

H = np.zeros((nocc * nvir, nocc * nvir))

for i in range(nocc):
    for a in range(nvir):
        ia = i * nvir + a
        for j in range(nocc):
            for b in range(nvir):
                jb = j * nvir + b

                # Діагональний член: (ε_a - ε_i) δ_ij δ_ab
                if i == j and a == b:
                    H[ia, jb] = mo_energy[nocc + a] - mo_energy[i]

                # Двоелектронні члени

                H[ia, jb] += 4.0 * eri_iajb[i, a, j, b]  # 4(ia|jb)
                H[ia, jb] -= eri_ijab[i, j, a, b]        # -(ij|ab)
                H[ia, jb] -= eri_ibja[i, b, j, a]        # -(ib|ja)

print(f"Матриця H побудована: {H.shape}")

# Розв'язуємо CPHF рівняння для кожної компоненти поля
# H · U^x = -g^x, де g^x - дипольна матриця
U = np.zeros((3, nocc, nvir))  # U[x, i, a] - коефіцієнти відгуку

print("\nРозв'язування CPHF рівнянь...")
for x in range(3):
    # Права частина: g^x[ia] = μ^x[ia] (occupied-virtual блок)
    g = -dip_ints_mo[x, :nocc, nocc:].flatten()

    # Розв'язуємо H · U = g
    U_flat = np.linalg.solve(H, g)
    U[x] = U_flat.reshape(nocc, nvir)

    print(f"  Компонента {'xyz'[x]}: ||U|| = {np.linalg.norm(U[x]):.6f}")

# Обчислюємо поляризованість: α_xy = -2·Σ_ia (μ^x_ia U^y_ia + μ^y_ia U^x_ia)
# Множник 2 від closed-shell (кожна МО має 2 електрони)
print("\nОбчислення тензора поляризованості...")
alpha = np.zeros((3, 3))

for x in range(3):
    for y in range(3):
        # α_xy = -2·Σ_{i,a} (μ^x_{ia} · U^y_{ia} + μ^y_{ia} · U^x_{ia})
        term1 = np.sum(dip_ints_mo[x, :nocc, nocc:] * U[y])
        term2 = np.sum(dip_ints_mo[y, :nocc, nocc:] * U[x])
        alpha[x, y] = -2.0 * (term1 + term2)

# Симетризація (повинна бути симетричною)
alpha = 0.5 * (alpha + alpha.T)

# Результати
print("\n" + "="*60)
print("РЕЗУЛЬТАТИ CPHF")
print("="*60)
print("\nТензор поляризованості α (a.u., 3×3):")
print(alpha)

print("\nДіагональні компоненти (α_xx, α_yy, α_zz):")
print(np.diag(alpha))

print("\nСередня поляризованість ⟨α⟩ = Tr(α)/3:")
print(f"{np.trace(alpha) / 3:.6f} a.u.")

# Порівняння з вбудованим методом PySCF
print("\n" + "="*60)
print("ПОРІВНЯННЯ З PYSCF")
print("="*60)
from pyscf.prop.polarizability.rhf import Polarizability
pol = Polarizability(mf)
alpha_pyscf = pol.polarizability()

print("\nТензор від PySCF:")
print(alpha_pyscf)

print("\nРізниця (CPHF - PySCF):")
diff = alpha - alpha_pyscf
print(diff)
print(f"\nМакс. абсолютна різниця: {np.max(np.abs(diff)):.3e}")

