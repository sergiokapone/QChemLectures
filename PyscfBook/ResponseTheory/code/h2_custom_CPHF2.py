import numpy as np
from pyscf import gto, scf
from pyscf.prop.polarizability.rhf import Polarizability

# === H2O ===
mol = gto.M(
    atom='''
    O 0.000000 0.000000 0.000000
    H 0.000000 -0.757000 0.587000
    H 0.000000 0.757000 0.587000
    ''',
    basis='6-31g',
    verbose=0
)

mf = scf.RHF(mol).run(conv_tol=1e-12)
print(f"Енергія: {mf.e_tot:.10f} Hartree")

mo_coeff = mf.mo_coeff
# mo_coeff — матриця коефіцієнтів молекулярних орбіталей (МО) у базисі АО
# Розмір: (nao, nmo), де nao — кількість атомних орбіталей, nmo — кількість МО
# mo_coeff[:, k] — k-та МО як лінійна комбінація АО

mo_energy = mf.mo_energy
# mo_energy — енергії молекулярних орбіталей (у Hartree)
# Розмір: (nmo,) — вектор енергій, відсортований за зростанням
# mo_energy[i] — енергія i-тої МО

mo_occ = mf.mo_occ
# mo_occ — окупації МО (0, 1 або 2 для RHF)
# Розмір: (nmo,) — 2.0 для зайнятих, 0.0 для віртуальних
# Для H₂O: перші 5 МО — зайняті (по 2 електрони), решта — віртуальні

nocc = int(np.sum(mo_occ > 0))
# nocc — кількість зайнятих МО
# mo_occ > 0 → True для всіх зайнятих → сума дає кількість
# int() — на випадок float (наприклад, 5.0 → 5)

nvir = len(mo_energy) - nocc
# nvir — кількість віртуальних МО
# len(mo_energy) = nmo = nocc + nvir → nvir = nmo - nocc

ov = nocc * nvir
# ov — загальна кількість occupied-virtual пар (ia)
# Розмірність задачі CPHF: ov × ov (матриця H), ov (вектор U)
# Для H₂O (6-31G): nocc=5, nvir=8 → ov=40

print("="*60)
print("CPHF ПОЛЯРИЗОВАНІСТЬ")
print("="*60)
print(f"nocc = {nocc}, nvir = {nvir}, ov = {ov}\n")

# === Диполь ===
with mol.with_common_orig((0,0,0)):
    dip_ao = mol.intor_symmetric('int1e_r', comp=3)
dip_mo = np.einsum('pi,xpq,qj->xij', mo_coeff, dip_ao, mo_coeff)

# === ERI (двоелектронні інтеграли) ===

C_occ = mo_coeff[:, :nocc]
# C_occ — коефіцієнти МО для зайнятих орбіталей
# Розмір: (nao, nocc) — стовпці: МО 0..nocc-1 (зайняті)
# Використовується для проєкції ERI на occupied підпростір

C_vir = mo_coeff[:, nocc:]
# C_vir — коефіцієнти МО для віртуальних орбіталей
# Розмір: (nao, nvir) — стовпці: МО nocc..nmo-1 (віртуальні)
# Використовується для проєкції ERI на virtual підпростір

eri_ao = mol.intor('int2e')
# eri_ao — двоелектронні інтеграли в базисі АО
# Розмір: (nao, nao, nao, nao) — eri_ao[p,q,r,s] = (pq|rs)
# Повний тензор, симетричний, обчислюється PySCF
# Далі буде трансформований у МО базис (ia|jb), (ij|ab) тощо

print("ERI → MO...")
# Вивід: початок трансформації двоелектронних інтегралів з AO → MO базису
# Потрібні блоки: (ia|jb), (ij|ab), (ib|ja) — для побудови гессіану CPHF

eri_iajb = np.einsum('pi,pqrs->iqrs', C_occ, eri_ao)
# Крок 1: проєкція по індексу p → i (occupied)
# eri_ao[p,q,r,s] → Σ_p C_occ[p,i] * (pqrs) → (i,q,r,s)
# Результат: (nocc, nao, nao, nao)

eri_iajb = np.einsum('qa,iqrs->iars', C_vir, eri_iajb)
# Крок 2: проєкція по індексу q → a (virtual)
# (i,q,r,s) → Σ_q C_vir[q,a] * (i,q,r,s) → (i,a,r,s)
# Результат: (nocc, nvir, nao, nao)

eri_iajb = np.einsum('rj,iars->iajs', C_occ, eri_iajb)
# Крок 3: проєкція по індексу r → j (occupied)
# (i,a,r,s) → Σ_r C_occ[r,j] * (i,a,r,s) → (i,a,j,s)
# Результат: (nocc, nvir, nocc, nao)

eri_iajb = np.einsum('sb,iajs->iajb', C_vir, eri_iajb)  # (i,a,j,b)
# Крок 4: проєкція по індексу s → b (virtual)
# (i,a,j,s) → Σ_s C_vir[s,b] * (i,a,j,s) → (i,a,j,b)
# Результат: (nocc, nvir, nocc, nvir) — блок (ia|jb)

eri_ijab = np.einsum('pi,pqrs->iqrs', C_occ, eri_ao)
# Початок (ij|ab): проєкція p → i
# (pqrs) → (i,q,r,s)

eri_ijab = np.einsum('qj,iqrs->ijrs', C_occ, eri_ijab)
# Проєкція q → j (occupied)
# (i,q,r,s) → (i,j,r,s)

eri_ijab = np.einsum('ra,ijrs->ijas', C_vir, eri_ijab)
# Проєкція r → a (virtual)
# (i,j,r,s) → (i,j,a,s)

eri_ijab = np.einsum('sb,ijas->ijab', C_vir, eri_ijab)  # (i,j,a,b)
# Проєкція s → b (virtual)
# Результат: (nocc, nocc, nvir, nvir) — блок (ij|ab)

eri_ibja = eri_iajb.transpose(0, 3, 2, 1)
# (ib|ja) = (ia|jb) з перестановкою a↔b
# (i,a,j,b) → (i,b,j,a) через transpose(0,3,2,1)
# Швидкий і точний спосіб отримати антисиметричний блок

# === H (електронний гессіан CPHF) ===
print("H матриця...")
# Вивід: початок побудови матриці гессіану H[ia,jb] розміром (ov × ov)
# H · U = g → розв’язок для коефіцієнтів відгуку U

i_occ = np.arange(nocc).repeat(nvir)
# i_occ — індекси зайнятих МО для всіх ov елементів
# Форма: [0,0,0,0, 1,1,1,1, ..., nocc-1,...] (довжина = ov)
# repeat(nvir) — кожне i повторюється nvir разів

a_vir = np.tile(np.arange(nvir), nocc)
# a_vir — індекси віртуальних МО для всіх ov елементів
# Форма: [0,1,2,...,nvir-1, 0,1,2,...,nvir-1, ...] (nocc повторів)
# tile(nocc) — кожен a повторюється nocc разів

H = np.diag(mo_energy[nocc + a_vir] - mo_energy[i_occ])
# Діагональний блок: (ε_a - ε_i) δ_ij δ_ab
# mo_energy[nocc + a_vir] — енергія віртуальної МО a
# mo_energy[i_occ] — енергія зайнятої МО i
# Результат: діагональна матриця (ov × ov) з різницею енергій

H += 4.0 * eri_iajb.reshape(ov, ov)
# Додаємо кулонівський внесок: +4(ia|jb)
# eri_iajb[i,a,j,b] → reshape → (ia, jb)
# Множник 4 — для закритої оболонки (2 електрони на МО)

H -= eri_ijab.transpose(0, 2, 1, 3).reshape(ov, ov)
# Віднімаємо обмінний внесок: -(ij|ab)
# eri_ijab[i,j,a,b] → transpose(0,2,1,3) → (i,a,j,b)
# Тепер індекси збігаються з (ia, jb) → reshape → додаємо до H

H -= eri_ibja.transpose(0, 3, 2, 1).reshape(ov, ov)
# Віднімаємо другий обмінний внесок: -(ib|ja)
# eri_ibja[i,b,j,a] → transpose(0,3,2,1) → (i,a,j,b)
# Тепер (ia, jb) → reshape → додаємо до H

print(f"H: {H.shape}")
# Вивід розміру матриці H: (ov, ov)
# Для H₂O: ov = 5×8 = 40 → H: (40, 40)

# === CPHF: розв'язання системи H U = g для всіх компонент поля (x, y, z) ===
g_ov = -dip_mo[:, :nocc, nocc:]
# g_ov — права частина CPHF рівнянь: g^x[ia] = -μ^x_ia
# Знак "-" — стандартна конвенція PySCF: H U = -μ_ov
# mu_ov = dip_mo[:, :nocc, nocc:] — дипольний блок (occ→vir)
# Результат: (3, nocc, nvir) — для кожної компоненти поля

U = np.zeros((3, nocc, nvir))
# U — матриця коефіцієнтів відгуку (пертурбації МО)
# U[x, i, a] — амплітуда змішування МО i → a під дією поля x
# Розмір: (3, nocc, nvir) — по одній матриці на компоненту

print("\nCPHF...")
# Вивід: початок розв'язання CPHF рівнянь

for x in range(3):
    # Для кожної компоненти поля (x=0,1,2 → x,y,z):
    U[x] = np.linalg.solve(H, g_ov[x].ravel()).reshape(nocc, nvir)
    # 1. g_ov[x].ravel() — вектор g^x розміром (ov,)
    # 2. np.linalg.solve(H, g) — розв'язок H U_flat = g → U_flat
    # 3. reshape(nocc, nvir) — повернення до форми (i,a)

    print(f"  {['x','y','z'][x]}: ||U|| = {np.linalg.norm(U[x]):.6f}")
    # Вивід норми ||U^x|| — міра величини відгуку на поле
    # Велика норма → сильна поляризованість у цьому напрямку

# === α_xy = –2 Σ (μ^x U^y + μ^y U^x) ===

# mu_ov — дипольні інтеграли між зайнятими (occ) та віртуальними (vir) МО:
# mu_ov[x, i, a] = <MO_i | μ^x | MO_a>  (x = 0,1,2 → x,y,z)
# Розмір: (3, nocc, nvir) — ключовий блок для CPHF (права частина рівнянь)
mu_ov = dip_mo[:, :nocc, nocc:]

# mu_ov: (3, nocc, nvir)
# U:     (3, nocc, nvir)

# Крок 1: Σ_{ia} μ^x_ia U^y_ia → einsum
# Результат: (3, 3)
term1 = np.einsum('xia,yia->xy', mu_ov, U)
# term1[x,y] = Σ_{i,a} μ^x_ia * U^y_ia
# Скалярний добуток ov-блоків диполя та відгуку
# 'xia' — μ^x (3, nocc, nvir), 'yia' — U^y (3, nocc, nvir)
# → сума по i,a → матриця (3,3)

term2 = np.einsum('yia,xia->xy', mu_ov, U)
# term2[x,y] = Σ_{i,a} μ^y_ia * U^x_ia
# Дзеркальний внесок (симетрія α_xy = α_yx)

# Крок 2: α_xy = -2 * (term1 + term2)
alpha = -2.0 * (term1 + term2)
# Множник -2:
#   • -1 — від знаку в правій частині CPHF (g = -μ_ov)
#   • ×2 — від закритої оболонки (кожна МО має 2 електрони)
# Результат: тензор поляризованості α_xy у атомних одиницях

# Крок 3: Симетризація (не обов’язково, але для точності)
alpha = 0.5 * (alpha + alpha.T)
# Забезпечує α_xy = α_yx через числові похибки
# У теорії тензор симетричний, на практиці — невелика асиметрія


# === Вивід ===
print("\n" + "="*60)
print("НАШ РЕЗУЛЬТАТ")
print(alpha.round(8))
print(f"⟨α⟩ = {np.trace(alpha)/3:.6f} a.u.")

# === PySCF ===
print("\n" + "="*60)
pol = Polarizability(mf)
alpha_ref = pol.polarizability()
print("PySCF:")
print(alpha_ref.round(8))
print("Різниця:")
print((alpha - alpha_ref).round(8))
print(f"max|Δ| = {np.max(np.abs(alpha - alpha_ref)):.2e}")

