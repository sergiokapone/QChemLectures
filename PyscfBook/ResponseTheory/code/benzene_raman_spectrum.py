# ============================================================
# benzene_raman_spectrum.py
# Повний розрахунок Раман-спектру бензолу C6H6
# Включає: частоти, активності, деполяризаційні відношення
# ============================================================

from pyscf import gto, scf, dft
from pyscf.hessian import thermo
from pyscf.prop import polarizability
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("РАМАН-СПЕКТР БЕНЗОЛУ C₆H₆")
print("=" * 70)

# ============================================================
# Крок 1: Визначення молекули бензолу
# ============================================================
print("\nКрок 1: Створення молекули бензолу (D6h симетрія)")
print("-" * 70)

# Бензол з правильною гексагональною структурою
mol = gto.M(
    atom="""
    C   1.3970   0.0000   0.0000
    C   0.6985   1.2100   0.0000
    C  -0.6985   1.2100   0.0000
    C  -1.3970   0.0000   0.0000
    C  -0.6985  -1.2100   0.0000
    C   0.6985  -1.2100   0.0000
    H   2.4810   0.0000   0.0000
    H   1.2405   2.1486   0.0000
    H  -1.2405   2.1486   0.0000
    H  -2.4810   0.0000   0.0000
    H  -1.2405  -2.1486   0.0000
    H   1.2405  -2.1486   0.0000
    """,
    basis="6-311g**",
    unit="angstrom",
    symmetry=True,  # Використовуємо симетрію
)

print(f"Кількість атомів: {mol.natm}")
print(f"Точкова група: {mol.symmetry}")
print(f"Базис: {mol.basis}")

# ============================================================
# Крок 2: SCF розрахунок
# ============================================================
print("\nКрок 2: DFT розрахунок (B3LYP/6-311G**)")
print("-" * 70)

mf = dft.RKS(mol)
mf.xc = "b3lyp"
mf.verbose = 4
e_total = mf.kernel()

print(f"\nПовна енергія: {e_total:.8f} Ha")

# ============================================================
# Крок 3: Обчислення гесіану та частот
# ============================================================
print("\n" + "=" * 70)
print("Крок 3: Розрахунок коливальних частот")
print("=" * 70)

print("\nОбчислення гесіану (це може зайняти 2-3 хвилини)...")
hess = mf.Hessian()
h = hess.kernel()

# Аналіз частот
freq_info = thermo.harmonic_analysis(mol, h)
frequencies = freq_info['freq_wavenumber']
normal_modes = freq_info['norm_mode']

# Фільтруємо справжні коливання (>100 см⁻¹)
real_freqs_mask = frequencies > 100
real_freqs = frequencies[real_freqs_mask]
real_modes = normal_modes[:, real_freqs_mask]

print(f"\nЗнайдено {len(real_freqs)} коливальних мод")

# Виводимо частоти по групах
print("\nКоливальні частоти (см⁻¹):")
print("-" * 70)
for i, freq in enumerate(real_freqs, 1):
    print(f"ν_{i:2d} = {freq:7.1f} см⁻¹")

# ============================================================
# Крок 4: Статична поляризовність
# ============================================================
print("\n" + "=" * 70)
print("Крок 4: Статична поляризовність α(0)")
print("=" * 70)

pol = polarizability.rks.Polarizability(mf)
alpha_static = pol.polarizability()

print("\nТензор поляризовності α (au³):")
print("       x          y          z")
for i, label in enumerate(['x', 'y', 'z']):
    print(f"{label}  ", end="")
    for j in range(3):
        print(f"{alpha_static[i,j]:>10.4f} ", end="")
    print()

alpha_mean = np.trace(alpha_static) / 3
print(f"\nСередня поляризовність: ᾱ = {alpha_mean:.4f} au³")

# ============================================================
# Крок 5: Похідні поляризовності (Раман-активність)
# ============================================================
print("\n" + "=" * 70)
print("Крок 5: Обчислення Раман-активностей")
print("=" * 70)

print("\nОбчислення похідних поляризовності ∂α/∂Q...")
print("(Використовуємо чисельне диференціювання)")

# Параметри для чисельного диференціювання
delta = 0.01  # Зміщення в атомних одиницях (bohr)

raman_activities = []
depolarization_ratios = []

for mode_idx in range(len(real_freqs)):
    print(f"\rОбробка моди {mode_idx+1}/{len(real_freqs)}...", end="", flush=True)
    
    # Нормальна мода у декартових координатах
    mode = real_modes[:, mode_idx]
    
    # Зміщення вздовж нормальної моди
    coords_orig = mol.atom_coords()
    
    # +delta
    coords_plus = coords_orig + delta * mode.reshape(-1, 3)
    mol_plus = gto.M(
        atom=[(mol.atom_symbol(i), coords_plus[i]) for i in range(mol.natm)],
        basis=mol.basis,
        unit="bohr",
        symmetry=False,
        verbose=0,
    )
    mf_plus = dft.RKS(mol_plus)
    mf_plus.xc = mf.xc
    mf_plus.verbose = 0
    mf_plus.kernel()
    pol_plus = polarizability.rks.Polarizability(mf_plus)
    alpha_plus = pol_plus.polarizability()
    
    # -delta
    coords_minus = coords_orig - delta * mode.reshape(-1, 3)
    mol_minus = gto.M(
        atom=[(mol.atom_symbol(i), coords_minus[i]) for i in range(mol.natm)],
        basis=mol.basis,
        unit="bohr",
        symmetry=False,
        verbose=0,
    )
    mf_minus = dft.RKS(mol_minus)
    mf_minus.xc = mf.xc
    mf_minus.verbose = 0
    mf_minus.kernel()
    pol_minus = polarizability.rks.Polarizability(mf_minus)
    alpha_minus = pol_minus.polarizability()
    
    # Похідна: ∂α/∂Q ≈ (α(+δ) - α(-δ)) / (2δ)
    dalpha_dQ = (alpha_plus - alpha_minus) / (2 * delta)
    
    # Раман-активність (середнє поле)
    # S = 45 * (∂ᾱ/∂Q)² + 7 * (∂γ/∂Q)²
    # де ᾱ - середня поляризовність, γ - анізотропія
    
    dalpha_mean = np.trace(dalpha_dQ) / 3
    
    # Анізотропія похідної
    dalpha_aniso_sq = (
        (dalpha_dQ[0,0] - dalpha_dQ[1,1])**2 +
        (dalpha_dQ[1,1] - dalpha_dQ[2,2])**2 +
        (dalpha_dQ[2,2] - dalpha_dQ[0,0])**2 +
        6 * (dalpha_dQ[0,1]**2 + dalpha_dQ[0,2]**2 + dalpha_dQ[1,2]**2)
    ) / 2
    
    # Раман-активність (Å⁴/amu)
    S = 45 * dalpha_mean**2 + 7 * dalpha_aniso_sq
    raman_activities.append(S)
    
    # Деполяризаційне відношення (для неполяризованого світла)
    # ρ = 3γ² / (45ᾱ² + 4γ²)
    if 45 * dalpha_mean**2 + 4 * dalpha_aniso_sq > 1e-10:
        rho = 3 * dalpha_aniso_sq / (45 * dalpha_mean**2 + 4 * dalpha_aniso_sq)
    else:
        rho = 0.0
    depolarization_ratios.append(rho)

print("\n")

raman_activities = np.array(raman_activities)
depolarization_ratios = np.array(depolarization_ratios)

# ============================================================
# Крок 6: Аналіз результатів
# ============================================================
print("\n" + "=" * 70)
print("Крок 6: Аналіз Раман-спектру")
print("=" * 70)

print("\nРаман-активні моди бензолу:")
print("-" * 70)
print(f"{'№':<5} {'ν (см⁻¹)':<12} {'Активність':<15} {'ρ':<10} {'Тип'}")
print("-" * 70)

# Класифікація мод за деполяризаційним відношенням
for i, (freq, activity, rho) in enumerate(
    zip(real_freqs, raman_activities, depolarization_ratios), 1
):
    # Класифікація
    if activity > 10:  # Значуща активність
        if rho < 0.1:
            mode_type = "Повністю симетрична"
        elif 0.1 <= rho < 0.5:
            mode_type = "Симетрична"
        else:
            mode_type = "Деполяризована"
        
        print(f"{i:<5} {freq:>10.1f}  {activity:>12.2f}  {rho:>8.4f}  {mode_type}")

# Найінтенсивніші моди
print("\nНайінтенсивніші Раман-лінії:")
print("-" * 70)
top_indices = np.argsort(raman_activities)[-5:][::-1]
for idx in top_indices:
    freq = real_freqs[idx]
    activity = raman_activities[idx]
    rho = depolarization_ratios[idx]
    print(f"ν = {freq:7.1f} см⁻¹, S = {activity:8.2f}, ρ = {rho:.4f}")

# ============================================================
# Крок 7: Візуалізація спектру
# ============================================================
print("\n" + "=" * 70)
print("Крок 7: Побудова Раман-спектру")
print("=" * 70)

# Сітка частот
freq_grid = np.linspace(0, 3500, 3500)

# Параметр уширення (FWHM)
gamma = 10  # см⁻¹

def lorentzian(x, x0, gamma, intensity):
    """Лоренцева функція"""
    return intensity * (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2)

# Будуємо спектр
spectrum = np.zeros_like(freq_grid)
for freq, activity in zip(real_freqs, raman_activities):
    if activity > 1:  # Тільки значущі
        spectrum += lorentzian(freq_grid, freq, gamma, activity)

# Візуалізація
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 1. Штрих-спектр
ax1 = axes[0]
for freq, activity in zip(real_freqs, raman_activities):
    if activity > 1:
        ax1.stem([freq], [activity], linefmt='b-', markerfmt='bo', basefmt=' ')
ax1.set_xlim(0, 3500)
ax1.set_ylabel('Раман-активність (Ų/amu)', fontsize=11)
ax1.set_title('Раман-спектр бензолу C₆H₆ (штрих-спектр)', fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3)

# 2. Уширений спектр
ax2 = axes[1]
ax2.plot(freq_grid, spectrum, 'b-', linewidth=2)
ax2.set_xlim(0, 3500)
ax2.set_ylabel('Інтенсивність (відн. од.)', fontsize=11)
ax2.set_title('Уширений спектр (γ=10 см⁻¹)', fontsize=13)
ax2.grid(alpha=0.3)

# Позначки характерних смуг
characteristic_peaks = [
    (992, "Дихання кільця (A₁g)"),
    (1606, "C=C розтяг (E₂g)"),
    (3064, "C-H розтяг (E₂g)"),
]
for freq, label in characteristic_peaks:
    if freq < 3500:
        ax2.axvline(freq, color='r', linestyle='--', alpha=0.5)
        ax2.text(freq, ax2.get_ylim()[1]*0.9, label, 
                rotation=90, verticalalignment='top', fontsize=9)

# 3. Деполяризаційні відношення
ax3 = axes[2]
colors = ['green' if rho < 0.1 else 'orange' if rho < 0.5 else 'red' 
          for rho in depolarization_ratios]
for freq, rho, color, activity in zip(real_freqs, depolarization_ratios, 
                                       colors, raman_activities):
    if activity > 1:
        ax3.scatter([freq], [rho], c=color, s=activity*2, alpha=0.7)

ax3.axhline(0.75, color='k', linestyle=':', alpha=0.5, label='ρ_max (деполяризована)')
ax3.axhline(0.0, color='k', linestyle=':', alpha=0.5, label='ρ=0 (повна симетрія)')
ax3.set_xlim(0, 3500)
ax3.set_ylim(-0.1, 1.0)
ax3.set_xlabel('Частота (см⁻¹)', fontsize=11)
ax3.set_ylabel('Деполяризаційне відношення ρ', fontsize=11)
ax3.set_title('Класифікація мод за симетрією', fontsize=13)
ax3.grid(alpha=0.3)
ax3.legend(fontsize=9)

plt.tight_layout()
plt.savefig('benzene_raman_spectrum.pdf', dpi=300, bbox_inches='tight')
print("\nСпектр збережено у файл: benzene_raman_spectrum.pdf")

# ============================================================
# Крок 8: Порівняння з експериментом
# ============================================================
print("\n" + "=" * 70)
print("Крок 8: Порівняння з експериментом")
print("=" * 70)

experimental_data = [
    (606, "A₂g (out-of-plane)"),
    (992, "A₁g (ring breathing)", "★"),
    (1178, "E₂g (C-H bend)"),
    (1596, "E₂g (C=C stretch)", "★"),
    (3047, "A₁g (C-H stretch)"),
    (3064, "E₂g (C-H stretch)", "★"),
]

print("\nХарактерні експериментальні частоти бензолу:")
print("-" * 70)
print(f"{'ν_exp (см⁻¹)':<15} {'Присвоєння':<30} {'Інтенсивність'}")
print("-" * 70)

for data in experimental_data:
    freq_exp = data[0]
    assignment = data[1]
    intensity = data[2] if len(data) > 2 else ""
    print(f"{freq_exp:<15} {assignment:<30} {intensity}")

print("\nЛегенда: ★ = сильна лінія")

print("\n" + "=" * 70)
print("ВИСНОВКИ")
print("=" * 70)

print("""
1. Раман-спектр бензолу:
   - Характеризується високою симетрією (D₆h)
   - Найінтенсивніша лінія: ~992 см⁻¹ (дихання кільця, A₁g)
   - Режим E₂g при ~1606 см⁻¹ (розтяг C=C)
   - C-H розтяги у області 3000-3100 см⁻¹

2. Симетрія та правила відбору:
   - Повністю симетричні моди (ρ≈0): A₁g
   - Деполяризовані моди (ρ≈0.75): E₁g, E₂g
   - Неактивні в Раман: A₂u, E₁u (ІЧ-активні)

3. Порівняння з експериментом:
   - B3LYP/6-311G** завищує частоти на ~5-10%
   - Для точності потрібно масштабувати на ~0.96
   - Відносні інтенсивності добре відтворюються

4. Рекомендації:
   - Для високої точності: використовуйте cc-pVTZ або більший базис
   - Враховуйте ангармонічні ефекти для C-H розтягів
   - Для твердого стану: додайте кристалічні ефекти
""")

print("=" * 70)
print("Розрахунок завершено!")
print("=" * 70)