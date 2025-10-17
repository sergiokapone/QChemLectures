# ============================================================
# plot_uv_spectrum.py
# Побудова УФ-спектру з уширенням
# ============================================================

from pyscf import gto, dft, tddft
import numpy as np
import matplotlib.pyplot as plt

mol = gto.M(
    atom="""
    C  0.0000  0.0000  0.0000
    O  0.0000  0.0000  1.2050
    H  0.0000  0.9428 -0.5876
    H  0.0000 -0.9428 -0.5876
    """,
    basis="aug-cc-pvdz",
    unit="angstrom",
)

# TDDFT розрахунок
mf = dft.RKS(mol)
mf.xc = "cam-b3lyp"
mf.verbose = 0
mf.kernel()

td = tddft.TDDFT(mf)
td.nstates = 10
td.verbose = 0
td.kernel()

# Отримуємо енергії та сили осциляторів
energies = td.e * 27.2114  # Ha -> eV
wavelengths = 1240 / energies  # eV -> nm
osc_strengths = td.oscillator_strength()

print("Побудова УФ-спектру")
print("=" * 60)
print("\nЗбудження:")
for i, (wl, f) in enumerate(zip(wavelengths, osc_strengths)):
    if f > 0.001:  # Тільки значущі переходи
        print(f"S_{i+1}: λ={wl:.1f} nm, f={f:.4f}")

# Будуємо спектр з гаусовим уширенням
def gaussian(x, center, sigma):
    return np.exp(-(x - center)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2*np.pi))

# Сітка довжин хвиль
wl_grid = np.linspace(150, 400, 1000)
spectrum = np.zeros_like(wl_grid)

# Параметр уширення
sigma = 10  # nm (для газової фази)

# Додаємо кожний перехід
for wl, f in zip(wavelengths, osc_strengths):
    if 150 < wl < 400:  # Тільки в УФ області
        spectrum += f * gaussian(wl_grid, wl, sigma)

# Малюємо
plt.figure(figsize=(10, 6))

# Штрих-спектр (окремі переходи)
plt.subplot(2, 1, 1)
for wl, f in zip(wavelengths, osc_strengths):
    if 150 < wl < 400 and f > 0.001:
        plt.stem([wl], [f], linefmt='b-', markerfmt='bo', basefmt=' ')
plt.xlim(150, 400)
plt.ylabel('Сила осцилятора')
plt.title('H₂CO: Штрих-спектр (TDDFT/CAM-B3LYP)')
plt.grid(alpha=0.3)

# Уширений спектр
plt.subplot(2, 1, 2)
plt.plot(wl_grid, spectrum, 'b-', linewidth=2)
plt.xlim(150, 400)
plt.xlabel('Довжина хвилі (нм)')
plt.ylabel('Інтенсивність (відн. од.)')
plt.title('Уширений спектр (σ=10 нм)')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('h2co_uv_spectrum.pdf', dpi=300, bbox_inches='tight')
print("\nСпектр збережено у файл h2co_uv_spectrum.pdf")

