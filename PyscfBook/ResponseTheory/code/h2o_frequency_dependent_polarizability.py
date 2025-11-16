# ============================================================
# h2o_frequency_dependent_polarizability.py
# Динамічна поляризовність (залежна від частоти)
# ============================================================

from pyscf import gto, scf
from pyscf.prop import polarizability
import numpy as np
import matplotlib.pyplot as plt

mol = gto.M(
    atom="""
    O  0.0000  0.0000  0.1173
    H  0.0000  0.7572 -0.4692
    H  0.0000 -0.7572 -0.4692
    """,
    basis="aug-cc-pvdz",
    unit="angstrom",
)

print("Частотно-залежна поляризовність H2O")
print("=" * 60)

# SCF розрахунок
mf = scf.RHF(mol)
mf.verbose = 0
mf.kernel()

# Частоти в au (1 au = 27.2114 eV)
# Розраховуємо від 0 до 10 eV
frequencies_ev = np.linspace(0, 10, 50)
frequencies_au = frequencies_ev / 27.2114

print("\nОбчислення поляризовності на різних частотах...")
print("(Це може зайняти хвилину)")

alpha_xx = []
alpha_yy = []
alpha_zz = []
alpha_mean = []

for freq in frequencies_au:
    # Динамічна поляризовність при частоті freq
    pol = polarizability.rhf.Polarizability(mf)
    alpha = pol.polarizability(freq=freq)

    alpha_xx.append(alpha[0, 0])
    alpha_yy.append(alpha[1, 1])
    alpha_zz.append(alpha[2, 2])
    alpha_mean.append(np.trace(alpha) / 3)

# Візуалізація
plt.figure(figsize=(10, 6))

plt.plot(frequencies_ev, alpha_xx, 'r-', label='αₓₓ', linewidth=2)
plt.plot(frequencies_ev, alpha_yy, 'g-', label='αᵧᵧ', linewidth=2)
plt.plot(frequencies_ev, alpha_zz, 'b-', label='αᵤᵤ', linewidth=2)
plt.plot(frequencies_ev, alpha_mean, 'k--', label='ᾱ (середня)', linewidth=2)

plt.xlabel('Енергія фотона (eV)', fontsize=12)
plt.ylabel('Поляризовність (au³)', fontsize=12)
plt.title('Частотна залежність поляризовності H₂O (RHF/aug-cc-pVDZ)', fontsize=13)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.xlim(0, 10)

plt.tight_layout()
plt.savefig('h2o_dynamic_polarizability.pdf', dpi=300)
print("\nГрафік збережено у файл h2o_dynamic_polarizability.pdf")

print("\nСтатичні значення (ω=0):")
print(f"  αₓₓ(0) = {alpha_xx[0]:.4f} au³")
print(f"  αᵧᵧ(0) = {alpha_yy[0]:.4f} au³")
print(f"  αᵤᵤ(0) = {alpha_zz[0]:.4f} au³")

print("\nПримітка:")
print("- Поляризовність зростає з частотою до резонансу")
print("- Різка зміна поблизу електронних збуджень (~7-8 eV)")
print("- Анізотропія також залежить від частоти")

