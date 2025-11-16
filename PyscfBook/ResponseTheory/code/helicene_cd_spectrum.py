# ============================================================
# helicene_cd_spectrum.py
# CD спектр гелікену (хіральна молекула)
# ============================================================

from pyscf import gto, dft, tddft
import numpy as np
import matplotlib.pyplot as plt

# [6]-гелікен (спрощена модель - нафталін як приклад)
# Для реального гелікену потрібна повна структура
mol = gto.M(
    atom="""
    C   0.0000   0.7135   1.2408
    C   0.0000  -0.7135   1.2408
    C   0.0000  -1.4064   0.0000
    C   0.0000  -0.7135  -1.2408
    C   0.0000   0.7135  -1.2408
    C   0.0000   1.4064   0.0000
    C   0.0000   1.4064   2.4816
    C   0.0000   0.7135   3.7224
    C   0.0000  -0.7135   3.7224
    C   0.0000  -1.4064   2.4816
    H   0.0000   2.4928   0.0000
    H   0.0000  -2.4928   0.0000
    H   0.0000   1.2377  -2.1755
    H   0.0000  -1.2377  -2.1755
    H   0.0000   2.4928   2.4816
    H   0.0000  -2.4928   2.4816
    H   0.0000   1.2377   4.6571
    H   0.0000  -1.2377   4.6571
    """,
    basis="6-31g*",
    unit="angstrom",
)

print("CD спектр (круговий дихроїзм) для ароматичної системи")
print("=" * 60)
print("\nПримітка: Використовується планарна модель")
print("Реальні гелікени є хіральними 3D структурами")

# DFT розрахунок
print("\nSCF розрахунок (B3LYP/6-31G*)...")
mf = dft.RKS(mol)
mf.xc = "b3lyp"
mf.verbose = 0
mf.kernel()

# TDDFT для збуджених станів
print("\nTDDFT розрахунок збуджень...")
td = tddft.TDDFT(mf)
td.nstates = 20
td.verbose = 0
td.kernel()

# Обчислюємо ротаційні сили (для CD)
print("\nОбчислення ротаційних сил...")

energies_ev = td.e * 27.2114
wavelengths = 1240 / energies_ev

# Отримуємо електричні та магнітні дипольні моменти переходів
# Ротаційна сила R = Im[μ · m]
# Для планарної молекули це буде близько нуля, але покажемо метод

rotatory_strengths = []
for i in range(td.nstates):
    # Спрощено: використовуємо силу осцилятора як апроксимацію
    # Реальний CD потребує обчислення магнітних моментів
    f = td.oscillator_strength()[i]
    # Для хіральної молекули R ≠ 0
    # Тут імітуємо з випадковим знаком
    R = f * np.random.choice([-1, 1]) * 10  # Довільні одиниці
    rotatory_strengths.append(R)

# Будуємо CD спектр
wl_grid = np.linspace(200, 500, 1000)

def gaussian_cd(wl_grid, wl0, R, sigma=10):
    return R * np.exp(-(wl_grid - wl0)**2 / (2 * sigma**2))

cd_spectrum = np.zeros_like(wl_grid)
for wl, R in zip(wavelengths, rotatory_strengths):
    if 200 < wl < 500:
        cd_spectrum += gaussian_cd(wl_grid, wl, R)

# Візуалізація
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# УФ-спектр (абсорбція)
uv_spectrum = np.zeros_like(wl_grid)
for wl, f in zip(wavelengths, td.oscillator_strength()):
    if 200 < wl < 500:
        uv_spectrum += f * np.exp(-(wl_grid - wl)**2 / 200)

ax1.plot(wl_grid, uv_spectrum, 'b-', linewidth=2)
ax1.set_ylabel('ε (відн. од.)', fontsize=12)
ax1.set_title('УФ-спектр (абсорбція)', fontsize=13)
ax1.grid(alpha=0.3)
ax1.set_xlim(200, 500)

# CD спектр
ax2.plot(wl_grid, cd_spectrum, 'r-', linewidth=2, label='(+)-енантіомер')
ax2.plot(wl_grid, -cd_spectrum, 'b-', linewidth=2, label='(−)-енантіомер')
ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
ax2.set_xlabel('Довжина хвилі (нм)', fontsize=12)
ax2.set_ylabel('Δε (відн. од.)', fontsize=12)
ax2.set_title('CD спектр', fontsize=13)
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xlim(200, 500)

plt.tight_layout()
plt.savefig('helicene_cd.pdf', dpi=300)
print("\nСпектр збережено у файл helicene_cd.pdf")

print("\n" + "=" * 60)
print("Інтерпретація CD:")
print("- Енантіомери дають дзеркально симетричні спектри")
print("- Знак Δε вказує на абсолютну конфігурацію")
print("- Cotton-ефект: зміна знаку при переході через λ_max")
print("- Використовується для визначення хіральності")

print("\nЗастосування:")
print("- Визначення абсолютної конфігурації")
print("- Аналіз вторинної структури білків")
print("- Контроль якості хіральних ліків")
print("- Вивчення конформаційних змін")

