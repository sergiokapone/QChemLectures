"""
Кривої потенційної енергії RHF для H2
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf

# Діапазон міжядерних відстаней (у борах)
distances = np.linspace(0.8, 8.0, 40)
energies_rhf = []

print("Розрахунок PES для H2 методом RHF")
print("=" * 60)
print(f"{'R (bohr)':<10} {'E_RHF (Ha)':<14}")
print("-" * 60)

for R in distances:
    # Створення молекули
    mol = gto.Mole()
    mol.atom = f"""
        H  0.0  0.0  0.0
        H  0.0  0.0  {R}
    """
    mol.basis = "cc-pvdz"
    mol.charge = 0
    mol.spin = 0
    mol.unit = "Bohr"
    mol.verbose = 0
    mol.build()

    # RHF розрахунок
    mf_rhf = scf.RHF(mol)
    mf_rhf.conv_tol = 1e-10
    E_rhf = mf_rhf.kernel()
    energies_rhf.append(E_rhf)

    print(f"{R:<10.2f} {E_rhf:<14.8f}")

energies_rhf = np.array(energies_rhf)

print("=" * 60)

# Аналіз результатів
idx_min_rhf = np.argmin(energies_rhf)
R_e_rhf = distances[idx_min_rhf]
E_min_rhf = energies_rhf[idx_min_rhf]

# Дисоціаційна енергія для двох атомів H
E_2H = 2 * (-0.5)  # 2×E(H)

print("\nРЕЗУЛЬТАТИ АНАЛІЗУ:")
print("=" * 60)
print("RHF:")
print(f"  R_e = {R_e_rhf:.3f} bohr = {R_e_rhf * 0.529177:.3f} Å")
print(f"  E(R_e) = {E_min_rhf:.6f} Ha")
print(f"  D_e = {E_2H - E_min_rhf:.6f} Ha = {(E_2H - E_min_rhf) * 27.2114:.3f} eV")
print(f"  E(R→∞) = {energies_rhf[-1]:.6f} Ha")
print(f"  Помилка дисоціації: {(energies_rhf[-1] - E_2H) * 27.2114:.3f} eV")
print(f"  Правильна дисоціація? {'✓' if abs(energies_rhf[-1] - E_2H) < 0.01 else '✗ (іонна: H⁺ + H⁻)'}")
print("=" * 60)

# Візуалізація RHF PES
plt.figure(figsize=(10,5))
plt.plot(distances, energies_rhf, "b-", linewidth=2.5, label="RHF (неправильна дисоціація)")
plt.axhline(E_2H, color="green", linestyle=":", linewidth=2, label=f"2×E(H) = {E_2H:.3f} Ha")
plt.plot(R_e_rhf, E_min_rhf, "bo", markersize=10, label=f"RHF мінімум: R={R_e_rhf:.2f}")
plt.xlabel("Міжядерна відстань R (bohr)")
plt.ylabel("Енергія E (Ha)")
plt.title("Крива потенційної енергії H₂: RHF")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11, loc="upper right")
plt.ylim([-1.2, -0.4])
plt.tight_layout()
plt.savefig("h2_rhf.pdf", dpi=300, bbox_inches="tight")
plt.show()

# Збереження даних
np.savez(
    "h2_rhf_pes.npz",
    distances=distances,
    E_rhf=energies_rhf,
)
print("Дані збережено: h2_rhf_pes.npz")

