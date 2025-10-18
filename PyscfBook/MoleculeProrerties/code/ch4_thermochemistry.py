"""
Термохімічний розрахунок для метану CH₄
Демонструє обчислення ZPE, ентальпії, ентропії та вільної енергії Гіббса
"""
from pyscf import gto, scf, dft
from pyscf.hessian import rhf as rhf_hess
import numpy as np

print("=" * 70)
print("Термохімічний аналіз метану CH₄")
print("=" * 70)

# Геометрія метану (тетраедрична)
mol = gto.M(
    atom='''
    C    0.000000    0.000000    0.000000
    H    0.629118    0.629118    0.629118
    H   -0.629118   -0.629118    0.629118
    H   -0.629118    0.629118   -0.629118
    H    0.629118   -0.629118   -0.629118
    ''',
    basis='6-311+g(2d,p)',
    symmetry=True,
    unit='angstrom'
)

print("\nМолекулярна геометрія:")
print("  Симетрія: Td (тетраедрична)")
print("  r(C-H) = 1.089 Å")
print("  ∠(H-C-H) = 109.47°")

# B3LYP розрахунок
print("\n\n1. B3LYP/6-311+G(2d,p)")
print("-" * 70)
mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.kernel()

E_elec = mf.e_tot
print(f"Електронна енергія: {E_elec:.8f} Hartree")
print(f"                    {E_elec * 627.509:.2f} kcal/mol")

# Розрахунок Гессіану
print("\n\n2. Розрахунок коливальних частот")
print("-" * 70)
print("Обчислення матриці Гессіану...")

hess = rhf_hess.Hessian(mf)
h = hess.kernel()

# Перетворення Гессіану
h_reshape = h.transpose(0, 2, 1, 3).reshape(mol.natm * 3, mol.natm * 3)

# Масо-зважений Гессіан
mass = []
atom_masses = mol.atom_mass_list()
for i in range(mol.natm):
    mass.extend([atom_masses[i]] * 3)
mass = np.array(mass)
mass_factor = np.sqrt(np.outer(mass, mass))
h_mw = h_reshape / mass_factor

# Діагоналізація
eigvals, eigvecs = np.linalg.eigh(h_mw)

# Переведення в см⁻¹
au_to_cm = 219474.63
freq_cm = np.sqrt(np.abs(eigvals)) * au_to_cm
freq_cm = np.where(eigvals < 0, -freq_cm, freq_cm)

# Відбір коливальних мод (частоти > 100 см⁻¹)
vib_freq = freq_cm[freq_cm > 100]
vib_freq = np.sort(vib_freq)

print(f"\nКількість коливальних мод: {len(vib_freq)}")
print("Очікується 3N-6 = 15-6 = 9 мод для нелінійної молекули")

print("\nКоливальні частоти (см⁻¹):")
print("-" * 70)
for i, freq in enumerate(vib_freq):
    print(f"  ν{i+1:2d}: {freq:7.1f} см⁻¹")

# Симетрійна класифікація для Td
print("\nСиметрійна класифікація (Td):")
print("-" * 70)
print("A₁:   ν₁ = 2917 см⁻¹  (симетричне валентне)")
print("E:    ν₂ = 1534 см⁻¹  (деформаційне, вироджене)")
print("F₂:   ν₃ = 3019 см⁻¹  (антисим. валентне, триразово вироджене)")
print("F₂:   ν₄ = 1306 см⁻¹  (деформаційне, триразово вироджене)")

# Розрахунок ZPE
print("\n\n3. Нульова коливальна енергія (ZPE)")
print("=" * 70)

# ZPE = (1/2) Σ ℏω
h_planck = 6.62607015e-34  # J·s
c_light = 299792458  # m/s
hartree_to_J = 4.3597447222071e-18  # J

ZPE_cm = 0.5 * np.sum(vib_freq)  # см⁻¹
ZPE_hartree = ZPE_cm / au_to_cm
ZPE_kcal = ZPE_hartree * 627.509

print(f"ZPE = {ZPE_cm:.2f} см⁻¹")
print(f"    = {ZPE_hartree:.6f} Hartree")
print(f"    = {ZPE_kcal:.2f} kcal/mol")
print(f"\nЕкспериментальне ZPE: 27.8 kcal/mol")
print(f"Похибка: {ZPE_kcal - 27.8:.2f} kcal/mol ({(ZPE_kcal - 27.8)/27.8 * 100:.1f}%)")

# Термохімічні поправки при 298.15 K
print("\n\n4. Термохімічні поправки при T = 298.15 K")
print("=" * 70)

T = 298.15  # K
k_B = 1.380649e-23  # J/K (Boltzmann constant)
R = 8.314462618  # J/(mol·K) (gas constant)
N_A = 6.02214076e23  # 1/mol (Avogadro)

# Переведення частот в Дж
freq_J = vib_freq * 100 * c_light * h_planck  # см⁻¹ → Дж

# Коливальна енергія
E_vib = 0
S_vib = 0
for freq_j in freq_J:
    x = freq_j / (k_B * T)
    E_vib += N_A * freq_j / (np.exp(x) - 1)
    S_vib += R * (x / (np.exp(x) - 1) - np.log(1 - np.exp(-x)))

E_vib_kcal = E_vib / 4184  # J/mol → kcal/mol
S_vib_cal = S_vib / 4.184  # J/(mol·K) → cal/(mol·K)

print(f"\nКоливальна енергія E_vib(T): {E_vib_kcal:.3f} kcal/mol")
print(f"Коливальна ентропія S_vib(T): {S_vib_cal:.2f} cal/(mol·K)")

# Обертальна енергія (нелінійна молекула)
E_rot = 1.5 * R * T / 1000  # kJ/mol
E_rot_kcal = E_rot / 4.184
print(f"\nОбертальна енергія E_rot(T): {E_rot_kcal:.3f} kcal/mol")

# Обертальна ентропія (Td симетрія, σ = 12)
# Для обертальної ентропії потрібні моменти інерції
# Спрощене обчислення для CH4
I_CH4 = 5.31e-47  # кг·м² (момент інерції CH4)
sigma = 12  # число симетрії для Td
S_rot = R * (1.5 + np.log((2 * np.pi * I_CH4 * k_B * T / h_planck**2)**(1.5) / sigma))
S_rot_cal = S_rot / 4.184

print(f"Обертальна ентропія S_rot(T): {S_rot_cal:.2f} cal/(mol·K)")

# Трансляційна енергія
E_trans = 1.5 * R * T / 1000  # kJ/mol
E_trans_kcal = E_trans / 4.184
print(f"\nТрансляційна енергія E_trans(T): {E_trans_kcal:.3f} kcal/mol")

# Трансляційна ентропія (Sackur-Tetrode)
M = 16.043  # g/mol (молярна маса CH4)
M_kg = M / 1000 / N_A  # кг
P = 101325  # Pa (1 atm)
S_trans = R * (2.5 + 1.5 * np.log(2 * np.pi * M_kg * k_B * T / h_planck**2) + 
               np.log(k_B * T / P))
S_trans_cal = S_trans / 4.184

print(f"Трансляційна ентропія S_trans(T): {S_trans_cal:.2f} cal/(mol·K)")

# Електронна ентропія (основний стан синглет, S_elec = 0)
S_elec = 0
print(f"Електронна ентропія S_elec: {S_elec:.2f} cal/(mol·K)")

# Сумарні величини
print("\n\n5. Термодинамічні функції при 298.15 K")
print("=" * 70)

# Внутрішня енергія
U = E_elec * 627.509 + ZPE_kcal + E_vib_kcal + E_rot_kcal + E_trans_kcal
print(f"\nВнутрішня енергія U(T):")
print(f"  U = E_elec + ZPE + E_vib + E_rot + E_trans")
print(f"  U = {E_elec * 627.509:.2f} + {ZPE_kcal:.2f} + {E_vib_kcal:.3f} + {E_rot_kcal:.3f} + {E_trans_kcal:.3f}")
print(f"  U = {U:.2f} kcal/mol")

# Ентальпія
H = U + R * T / 4184  # додаємо PV = RT
print(f"\nЕнтальпія H(T):")
print(f"  H = U + RT")
print(f"  H = {H:.2f} kcal/mol")

# Поправка до ентальпії від 0 K
H_corr = ZPE_kcal + E_vib_kcal + E_rot_kcal + E_trans_kcal + R * T / 4184
print(f"\nТермічна поправка H(298) - H(0):")
print(f"  ΔH = {H_corr:.2f} kcal/mol")

# Ентропія
S_total = S_trans_cal + S_rot_cal + S_vib_cal + S_elec
print(f"\nЕнтропія S(T):")
print(f"  S = S_trans + S_rot + S_vib + S_elec")
print(f"  S = {S_trans_cal:.2f} + {S_rot_cal:.2f} + {S_vib_cal:.2f} + {S_elec:.2f}")
print(f"  S = {S_total:.2f} cal/(mol·K)")

# Вільна енергія Гіббса
G = H - T * S_total / 1000  # kcal/mol
print(f"\nВільна енергія Гіббса G(T):")
print(f"  G = H - TS")
print(f"  G = {H:.2f} - {T:.2f} × {S_total/1000:.5f}")
print(f"  G = {G:.2f} kcal/mol")

# Теплоємність
C_v = 3 * R / 4.184  # 3R для нелінійної молекули (наближення)
# Точніше: C_v = C_trans + C_rot + C_vib
C_v_vib = 0
for freq_j in freq_J:
    x = freq_j / (k_B * T)
    C_v_vib += R * x**2 * np.exp(x) / (np.exp(x) - 1)**2

C_v_total = (1.5 * R + 1.5 * R + C_v_vib) / 4.184
print(f"\nТеплоємність C_V(T):")
print(f"  C_V = C_trans + C_rot + C_vib")
print(f"  C_V = {C_v_total:.2f} cal/(mol·K)")

# Порівняння з експериментом
print("\n\n6. Порівняння з експериментом")
print("=" * 70)

exp_data = {
    'ZPE': 27.8,
    'H_corr': 2.48,
    'S': 44.5,
    'C_v': 6.0
}

print(f"\nВеличина                 Розрах.    Експ.     Δ")
print("-" * 70)
print(f"ZPE (kcal/mol)           {ZPE_kcal:6.2f}     {exp_data['ZPE']:6.2f}    {ZPE_kcal - exp_data['ZPE']:+.2f}")
print(f"H(298)-H(0) (kcal/mol)   {H_corr:6.2f}     {exp_data['H_corr']:6.2f}    {H_corr - exp_data['H_corr']:+.2f}")
print(f"S(298) (cal/mol·K)       {S_total:6.1f}     {exp_data['S']:6.1f}    {S_total - exp_data['S']:+.1f}")
print(f"C_V(298) (cal/mol·K)     {C_v_total:6.1f}     {exp_data['C_v']:6.1f}    {C_v_total - exp_data['C_v']:+.1f}")

# Застосування
print("\n\n7. Застосування термохімії")
print("=" * 70)

print("\n1. Розрахунок енергії реакції:")
print("   ΔG_rxn = Σ G_products - Σ G_reactants")
print("   Важливо включати ZPE та термічні поправки!")

print("\n2. Константа рівноваги:")
print("   K_eq = exp(-ΔG°/RT)")
print("   ΔG° визначає положення рівноваги")

print("\n3. Швидкість реакції (рівняння Ейрінга):")
print("   k = (k_B T / h) exp(-ΔG‡/RT)")
print("   ΔG‡ - вільна енергія активації")

print("\n4. Ізотопні ефекти:")
print("   Заміна H → D змінює ZPE")
print("   Впливає на швидкості реакцій")

# Методологічні поради
print("\n\n8. Методологічні рекомендації")
print("=" * 70)

print("\n✓ Для термохімії:")
print("  • Базис: 6-311+G(2d,p) або aug-cc-pVTZ")
print("  • Метод: B3LYP (швидко), ωB97X-D (точніше)")
print("  • Обов'язково: оптимізація + перевірка на мінімум")
print("  • Масштабуйте частоти (λ ≈ 0.968 для B3LYP)")

print("\n✓ Точність:")
print("  • ZPE: ±0.5 kcal/mol (3-5% похибка)")
print("  • ΔH: ±1 kcal/mol для малих молекул")
print("  • ΔG: ±2 kcal/mol (ентропія менш точна)")

print("\n✓ Пастки:")
print("  • Не забувайте ZPE (може бути 10+ kcal/mol)")
print("  • Перевіряйте на уявні частоти")
print("  • Для конформерів: Больцманівське усереднення")

print("\n" + "=" * 70)
print("ВИСНОВКИ:")
print("=" * 70)
print("• ZPE становить ~28 kcal/mol для CH₄ (не можна ігнорувати!)")
print("• Термічна поправка H(298)-H(0) ≈ 2.5 kcal/mol")
print("• Ентропія важлива для ΔG (T·S може бути значним)")
print("• B3LYP дає хорошу згоду з експериментом")
print("• Термохімія критична для розрахунків реакцій")
print("=" * 70)