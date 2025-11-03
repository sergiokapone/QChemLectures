# ============================================================
# h2o_ir_spectrum.py
# Розрахунок ІЧ-спектру (частоти + інтенсивності)
# ============================================================

import numpy as np
from pyscf import gto, scf
from pyscf.hessian import rhf as rhf_hess
from pyscf.hessian import thermo
import matplotlib.pyplot as plt

# ============================================================
# Створення молекули H2O
# ============================================================
mol = gto.M(
    atom="""
    O  0.0000  0.0000  0.1173
    H  0.0000  0.7572 -0.4692
    H  0.0000 -0.7572 -0.4692
    """,
    basis="6-31g",
    unit="angstrom",
)

print("=" * 70)
print("Розрахунок ІЧ-спектру H2O (частоти + інтенсивності)")
print("=" * 70)

# ============================================================
# SCF розрахунок
# ============================================================
print("\n[1/4] SCF розрахунок...")
mf = scf.RHF(mol)
mf.kernel()
print(f"Енергія: {mf.e_tot:.8f} Ha")

# ============================================================
# Обчислення Гесіану
# ============================================================
print("\n[2/4] Обчислення Гесіану...")
hess = mf.Hessian().kernel()

# ============================================================
# Аналіз нормальних мод
# ============================================================
print("\n[3/4] Аналіз нормальних мод...")
freq_info = thermo.harmonic_analysis(mol, hess)

frequencies = freq_info['freq_wavenumber']  # в см^-1
normal_modes = freq_info['norm_mode']  # власні вектори

# ============================================================
# Обчислення інтенсивностей ІЧ
# ============================================================
print("\n[4/4] Обчислення інтенсивностей ІЧ...")

# Крок для числового диференціювання дипольного моменту
h = 0.001  # Bohr

natm = mol.natm
intensities = []

for k in range(len(frequencies)):
    # Нормальна мода (3*natm вектор)
    L_k = normal_modes[:, k].reshape(natm, 3)

    # Зміщення вздовж нормальної моди
    R_0 = mol.atom_coords()  # в Bohr

    # Позитивне зміщення
    R_pos = R_0 + h * L_k
    mol_pos = gto.M(
        atom=[[mol.atom_symbol(i), R_pos[i]] for i in range(natm)],
        basis=mol.basis,
        unit='Bohr'
    )
    mf_pos = scf.RHF(mol_pos)
    mf_pos.verbose = 0
    mf_pos.kernel()
    dip_pos = mf_pos.dip_moment(unit='Bohr')  # атомні одиниці

    # Негативне зміщення
    R_neg = R_0 - h * L_k
    mol_neg = gto.M(
        atom=[[mol.atom_symbol(i), R_neg[i]] for i in range(natm)],
        basis=mol.basis,
        unit='Bohr'
    )
    mf_neg = scf.RHF(mol_neg)
    mf_neg.verbose = 0
    mf_neg.kernel()
    dip_neg = mf_neg.dip_moment(unit='AU')

    # Похідна дипольного моменту: dμ/dQ
    dip_derivative = (dip_pos - dip_neg) / (2 * h)

    # Інтенсивність: I ∝ |dμ/dQ|^2
    # В одиницях: (Debye/Angstrom)^2 -> km/mol
    # Константа: 974.8638 для конвертації
    intensity = np.linalg.norm(dip_derivative)**2 * 974.8638
    intensities.append(intensity)

intensities = np.array(intensities)

# ============================================================
# Виведення результатів
# ============================================================
print("\n" + "=" * 70)
print("ІЧ-СПЕКТР H2O")
print("=" * 70)
print(f"{'Мода':<8} {'ν (см⁻¹)':<15} {'Інтенсивність (km/mol)':<25} {'Тип'}")
print("-" * 70)

for i, (freq, intens) in enumerate(zip(frequencies, intensities)):
    if freq > 100:  # Тільки справжні коливання (не трансляції/обертання)
        mode_type = "сильна" if intens > 100 else ("середня" if intens > 10 else "слабка")
        print(f"{i+1:<8} {freq:>12.1f} {intens:>20.1f}     {mode_type}")

# ============================================================
# Візуалізація спектру
# ============================================================
print("\n[Візуалізація] Побудова ІЧ-спектру...")

# Фільтруємо тільки справжні коливання
real_modes = frequencies > 100
freq_plot = frequencies[real_modes]
intens_plot = intensities[real_modes]

# Створюємо спектр з гауссовим розширенням
x = np.linspace(0, 4500, 2000)
y = np.zeros_like(x)
fwhm = 20  # Full Width at Half Maximum (см^-1)

for freq, intens in zip(freq_plot, intens_plot):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    y += intens * np.exp(-((x - freq)**2) / (2 * sigma**2))

# Побудова графіку
plt.figure(figsize=(12, 6))
plt.plot(x, y, 'b-', linewidth=1.5)
plt.fill_between(x, y, alpha=0.3)

# Додаємо вертикальні лінії для піків
for freq, intens in zip(freq_plot, intens_plot):
    plt.axvline(freq, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
    plt.text(freq, intens * 1.05, f'{freq:.0f}',
             rotation=90, va='bottom', ha='right', fontsize=8)

plt.xlabel('Хвильове число (см⁻¹)', fontsize=12)
plt.ylabel('Інтенсивність (km/mol)', fontsize=12)
plt.title('ІЧ-спектр H₂O (RHF/6-31G)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim(0, 4500)
plt.ylim(0, max(y) * 1.2)

plt.tight_layout()
plt.savefig('h2o_ir_spectrum.png', dpi=300, bbox_inches='tight')
print("Спектр збережено в 'h2o_ir_spectrum.png'")
plt.show()

# ============================================================
# Інтерпретація мод
# ============================================================
print("\n" + "=" * 70)
print("ІНТЕРПРЕТАЦІЯ КОЛИВАЛЬНИХ МОД")
print("=" * 70)
