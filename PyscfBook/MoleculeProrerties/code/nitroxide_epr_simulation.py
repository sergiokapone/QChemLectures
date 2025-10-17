# ============================================================
# nitroxide_epr_simulation.py
# Симуляція ЕПР спектру нітроксидного радикала
# ============================================================

from pyscf import gto, scf, dft
from pyscf.prop.gtensor import uks as gtensor_uks
from pyscf.prop.hfc import uks as hfc_uks
import numpy as np
import matplotlib.pyplot as plt

# TEMPO (2,2,6,6-тетраметилпіперидин-1-оксил)
# Спрощена модель: використовуємо N-O фрагмент
mol = gto.M(
    atom="""
    N   0.0000   0.0000   0.0000
    O   0.0000   0.0000   1.2500
    C  -1.2500  -0.5000  -0.5000
    C   1.2500  -0.5000  -0.5000
    H  -2.0000  -0.5000   0.2500
    H  -1.5000   0.0000  -1.4000
    H   2.0000  -0.5000   0.2500
    H   1.5000   0.0000  -1.4000
    """,
    basis="6-311g**",
    spin=1,  # Нітроксил - дублет
    symmetry=False,
)

print("ЕПР спектр нітроксидного радикала")
print("=" * 60)

# UKS розрахунок
print("\nUKS розрахунок (B3LYP/6-311G**)...")
mf = dft.UKS(mol)
mf.xc = "b3lyp"
mf.verbose = 0
mf.kernel()

# g-тензор
print("\nОбчислення g-тензора...")
gtensor = gtensor_uks.GTensor(mf)
g = gtensor.kernel()
g_vals = np.linalg.eigvalsh(g)
g_iso = np.mean(g_vals)

print(f"\ng-фактори:")
print(f"  g_xx = {g_vals[0]:.6f}")
print(f"  g_yy = {g_vals[1]:.6f}")
print(f"  g_zz = {g_vals[2]:.6f}")
print(f"  g_iso = {g_iso:.6f}")

# Константи НТВ (особливо для ¹⁴N)
print("\nОбчислення констант НТВ...")
hfc = hfc_uks.HFC(mf)
a_iso, a_dip = hfc.kernel()

# Знаходимо азот (атом 0)
N_idx = 0
a_N_iso = a_iso[N_idx] * 1420.4057 * 0.403  # au -> МГц для ¹⁴N
a_N_tensor = a_dip[N_idx] * 1420.4057 * 0.403

print(f"\nКонстанта НТВ для ¹⁴N (I=1):")
print(f"  A_iso = {a_N_iso:.2f} МГц")
print(f"  A_xx = {a_N_tensor[0,0]:.2f} МГц")
print(f"  A_yy = {a_N_tensor[1,1]:.2f} МГц")
print(f"  A_zz = {a_N_tensor[2,2]:.2f} МГц")

# Симуляція спектру
print("\nСимуляція ЕПР спектру...")

# Параметри
B_range = np.linspace(3300, 3500, 1000)  # Gauss
linewidth = 2.0  # Gauss

# Лоренцева функція
def lorentzian(x, x0, gamma):
    return gamma**2 / ((x - x0)**2 + gamma**2)

# Резонансні поля для триплету від ¹⁴N (I=1, m_I = -1, 0, +1)
# B_res = (h*nu) / (g * mu_B) ± A * m_I
nu = 9.5  # ГГц (X-band ЕПР)
h_ghz = 0.71448  # планк в GHz/G
B0 = (h_ghz * nu) / g_iso  # Центральне поле

# Три лінії триплету
A_gauss = a_N_iso / 2.8025  # МГц -> Gauss
B_minus = B0 - A_gauss
B_center = B0
B_plus = B0 + A_gauss

# Спектр (абсорбція)
spectrum = (lorentzian(B_range, B_minus, linewidth) +
            lorentzian(B_range, B_center, linewidth) +
            lorentzian(B_range, B_plus, linewidth))

# Перша похідна (типовий ЕПР спектр)
spectrum_deriv = np.gradient(spectrum, B_range[1] - B_range[0])

# Візуалізація
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Абсорбція
ax1.plot(B_range, spectrum, 'b-', linewidth=2)
ax1.axvline(B_minus, color='r', linestyle='--', alpha=0.5, label='m_I=-1')
ax1.axvline(B_center, color='g', linestyle='--', alpha=0.5, label='m_I=0')
ax1.axvline(B_plus, color='r', linestyle='--', alpha=0.5, label='m_I=+1')
ax1.set_ylabel('Інтенсивність поглинання', fontsize=12)
ax1.set_title('ЕПР спектр нітроксиду: абсорбція', fontsize=13)
ax1.legend()
ax1.grid(alpha=0.3)

# Перша похідна
ax2.plot(B_range, spectrum_deriv, 'b-', linewidth=2)
ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
ax2.set_xlabel('Магнітне поле (Gauss)', fontsize=12)
ax2.set_ylabel('dI/dB', fontsize=12)
ax2.set_title('ЕПР спектр: перша похідна (експериментальний вигляд)', fontsize=13)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('nitroxide_epr_spectrum.pdf', dpi=300)
print("\nСпектр збережено у файл nitroxide_epr_spectrum.pdf")

print("\n" + "=" * 60)
print("Характеристики спектру:")
print(f"  Триплет від ¹⁴N (I=1)")
print(f"  Розщеплення A ≈ {A_gauss:.1f} Gauss")
print(f"  g_iso ≈ {g_iso:.6f}")
print("\nТиповий нітроксид (TEMPO):")
print("  A(¹⁴N) ≈ 15-17 Gauss")
print("  g ≈ 2.006")

