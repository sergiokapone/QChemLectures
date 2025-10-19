import numpy as np
from pyscf import gto, scf
from pyscf.hessian import thermo
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


# ============================================================
# ЕТАП 1: ГРУБИЙ ПОШУК МАКСИМУМУ ЕНЕРГІЇ
# ============================================================
def energy_func(h):
    """Энергия системы при высоте N = h над плоскостью H₃"""
    mol = gto.M(
        atom=f'''
        N   0.0000   0.0000   {h}
        H   0.0000   1.0124   0.0000
        H   0.8770  -0.5062   0.0000
        H  -0.8770  -0.5062   0.0000
        ''',
        basis='6-31g**',
        unit='angstrom'
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    return mf.kernel()


print("=" * 70)
print("АНАЛІЗ ІНВЕРСІЇ NH₃")
print("=" * 70)

# ============================================================
# ЕТАП 2: ДЕТАЛЬНИЙ АНАЛІЗ ІЗ ГЕССІАНОМ І ЧАСТОТАМИ
# ============================================================
def analyze_point(h, verbose=True):
    """Повний аналіз точки: енергія, градієнт, гессіан, частоти"""
    mol = gto.M(
        atom=f'''
        N   0.0000   0.0000   {h}
        H   0.0000   1.0124   0.0000
        H   0.8770  -0.5062   0.0000
        H  -0.8770  -0.5062   0.0000
        ''',
        basis='6-31g**',
        unit='angstrom',
        symmetry=False
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    e = mf.kernel()

    # Градиент
    grad = mf.Gradients().kernel()
    grad_signed = grad[0, 2]   # градиент по z атома N

    # Гессиан и частоты
    if verbose:
        print(f"  Вычисление гессиана для h = {h:.4f} Å...")
    hess = mf.Hessian().kernel()
    freq_info = thermo.harmonic_analysis(mol, hess, exclude_trans=True, imaginary_freq=True)
    frequencies = freq_info['freq_wavenumber']

    # Подсчёт мнимых частот
    imag_freqs = [f for f in frequencies if np.iscomplex(f)]
    num_imag = len(imag_freqs)

    return {
        'h': h,
        'energy': e,
        'grad_signed': grad_signed,
        'num_imag': num_imag,
        'imag_freqs': imag_freqs,
        'all_freqs': frequencies
    }

# Груба сітка по всьому діапазону
h_coarse = np.linspace(-0.25, 0.25, 10)

# Уточнення поблизу мінімумів (~ ±0.215 Å)
h_fine_minima = np.linspace(-0.23, -0.20, 15)
h_fine_minima2 = np.linspace(0.20, 0.23, 15)

# Уточнення поблизу перехідного стану (h ≈ 0)
h_fine_ts = np.linspace(-0.02, 0.02, 15)

# Об'єднуємо все разом
extra_points = [0.0, -0.215, 0.215]
h_detailed = np.unique(np.sort(np.concatenate([
    h_coarse, h_fine_minima, h_fine_minima2, h_fine_ts, extra_points
])))

results = []

print(f"\n{'h (Å)':<10} {'E_rel (ккал/моль)':<18} {'grad E':<12} {'Частоти (см⁻¹)':<50}")
print("-" * 100)

grad_tol = 1e-4      # поріг малості градієнта
imag_tol = 1e-6      # поріг для "нульової" уявної частини

results = []

for h in h_detailed:
    res_point = analyze_point(h, verbose=False)
    results.append(res_point)

    # енергія відносно мінімуму (за всіма вже порахованими)
    e_rel = (res_point['energy'] - min(r['energy'] for r in results)) * 627.51

    grad_signed = res_point['grad_signed']

    # показуємо частоти, якщо градієнт малий
    show_freq = abs(grad_signed) < grad_tol

    if show_freq:
        freqs_raw = np.array(res_point['all_freqs'], dtype=complex)
        freq_items = []
        for f in freqs_raw:
            if np.iscomplexobj(f) and abs(f.imag) > imag_tol:
                val = abs(f.imag) if abs(f.real) < imag_tol else np.sqrt(f.real**2 + f.imag**2)
                freq_items.append(f"i{val:.1f}")
            else:
                fr = float(np.real(f))
                if fr < 0:
                    freq_items.append(f"i{abs(fr):.1f}")
                else:
                    freq_items.append(f"{fr:.1f}")
        freq_display = ", ".join(freq_items)
    else:
        freq_display = ""

    print(f"{h:<10.4f} {e_rel:<18.4f} {grad_signed:<12.2e} {freq_display:<60}")


# ============================================================
# АНАЛІЗ ПЕРЕХІДНОГО СТАНУ
# ============================================================
print("\n" + "=" * 70)
print("АНАЛІЗ ПЕРЕХІДНОГО СТАНУ")
print("=" * 70)

# Шукаємо точку з максимальною енергією та мінімальною енергією
energies = [r['energy'] for r in results]
max_idx = np.argmax(energies)
min_idx = np.argmin(energies)
e_min = energies[min_idx]
ts_candidate = results[max_idx]

print(f"\nКандидат на TS: h = {ts_candidate['h']:.6f} Å")
print(f"  Відносна енергія: {(ts_candidate['energy'] - e_min) * 627.51:.4f} ккал/моль")
print(f"  Норма градієнта: {ts_candidate['grad_signed']:.2e}")
print(f"  Число уявних частот: {ts_candidate['num_imag']}")

# Перевірка критеріїв TS
is_ts = True
issues = []

if abs(ts_candidate['h']) > 0.02:
    is_ts = False
    issues.append(f"h не близьке до 0 (h = {ts_candidate['h']:.4f} Å)")

if ts_candidate['grad_signed'] > 5e-3:
    is_ts = False
    issues.append(f"Градієнт занадто великий ({ts_candidate['grad_signed']:.2e})")

if ts_candidate['num_imag'] != 1:
    is_ts = False
    issues.append(f"Не 1 уявна частота (знайдено {ts_candidate['num_imag']})")

if is_ts:
    print("\n✓✓✓ ЦЕ ПЕРЕХІДНИЙ СТАН! ✓✓✓")
    imag_freq = abs(ts_candidate['imag_freqs'][0])
    print(f"\n  Уявна частота інверсії: i{imag_freq:.2f} см⁻¹")
    print(f"  Бар'єр інверсії: {(ts_candidate['energy'] - e_min) * 627.51:.4f} ккал/моль")
else:
    print("\n❌ НЕ ПЕРЕХІДНИЙ СТАН")
    print("\пПроблеми:")
    for issue in issues:
        print(f"  • {issue}")


# ============================================================
# ВІЗУАЛІЗАЦІЯ
# ============================================================
fig, ax1 = plt.subplots(1, 1, figsize=(14, 5))

# Графік профілю енергії
h_vals = [r['h'] for r in results]
e_vals = [(r['energy'] - e_min) * 627.51 for r in results]
ax1.plot(h_vals, e_vals, 'r-', linewidth=2, label='Профіль енергії')

ax1.axvline(ts_candidate['h'], color='orange', linestyle=':', linewidth=2,
            label=f"Максимальна енергія: h={ts_candidate['h']:.3f} Å")
ax1.set_xlabel('Висота N над площиною H₃ (Å)', fontsize=11)
ax1.set_ylabel('Відносна енергія (ккал/моль)', fontsize=11)
ax1.set_title('Профіль енергії інверсії NH₃', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nh3_inversion_ts_full.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Графік збережено: nh3_inversion_ts_full.pdf")


# ============================================================
# ПІДСУМКОВЕ ЗВЕДЕННЯ
# ============================================================

print(f"\nПерехідний стан (з аналізу):")
print(f"  h(N) = {ts_candidate['h']:.6f} Å")
print(f"  Барьер = {(ts_candidate['energy'] - e_min) * 627.51:.4f} ккал/моль")
if ts_candidate['num_imag'] > 0:
    print(f"  Уявна частота = i{abs(ts_candidate['imag_freqs'][0]):.2f} см⁻¹")
print("=" * 70)

