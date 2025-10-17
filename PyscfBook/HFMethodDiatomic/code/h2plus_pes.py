"""
Побудова кривої потенційної енергії (PES) для H2+
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf

# Діапазон міжядерних відстаней (у борах)
distances = np.linspace(0.5, 5.0, 30)  # від 0.5 до 5.0 bohr
energies = []

print("Розрахунок PES для H2+...")
print("=" * 60)

for R in distances:
    # Створюємо молекулу з новою геометрією
    mol = gto.Mole()
    mol.atom = f"""
        H  0.0  0.0  0.0
        H  0.0  0.0  {R}
    """
    mol.basis = "cc-pvdz"  # Кращий базис для точності
    mol.charge = 1
    mol.spin = 1
    mol.unit = "Bohr"
    mol.verbose = 0  # Вимкнути детальний вивід
    mol.build()

    # UHF розрахунок
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-8
    E = mf.kernel()
    energies.append(E)

    print(f"R = {R:5.2f} bohr  →  E = {E:10.6f} Ha")

energies = np.array(energies)

# Знаходження мінімуму
idx_min = np.argmin(energies)
R_e = distances[idx_min]
E_min = energies[idx_min]

# Енергія дисоціації
E_dissoc = -0.5  # E(H) = -0.5 Ha
D_e = E_dissoc - E_min

print("=" * 60)
print(f"Рівноважна відстань R_e = {R_e:.3f} bohr = {R_e * 0.529:.3f} Å")
print(f"Енергія в мінімумі E_min = {E_min:.6f} Ha")
print(f"Енергія дисоціації D_e = {D_e:.6f} Ha = {D_e * 27.211:.3f} eV")
print("Експериментальне D_e ≈ 2.79 eV")
print("=" * 60)

# Побудова графіка
plt.figure(figsize=(10, 6))
plt.plot(distances, energies, "b-", linewidth=2, label="H$_2^+$ PES (cc-pVDZ)")
plt.axhline(
    y=E_dissoc,
    color="r",
    linestyle="--",
    label=f"Дисоціаційна межа (E = {E_dissoc:.3f} Ha)",
)
plt.plot(R_e, E_min, "go", markersize=10, label=f"Рівновага: R$_e$ = {R_e:.2f} bohr")

plt.xlabel("Міжядерна відстань R (bohr)", fontsize=12)
plt.ylabel("Енергія E (Ha)", fontsize=12)
plt.title("Крива потенційної енергії H$_2^+$", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("h2plus_pes.png", dpi=300, bbox_inches="tight")
print("\nГрафік збережено: h2plus_pes.png")
plt.show()

# Збереження даних
np.savez(
    "h2plus_pes_data.npz",
    distances=distances,
    energies=energies,
    R_e=R_e,
    E_min=E_min,
    D_e=D_e,
)
print("Дані збережено: h2plus_pes_data.npz")
