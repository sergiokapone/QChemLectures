"""
Порівняння кривих потенційної енергії RHF vs UHF для H2
Демонстрація проблеми дисоціації
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf

# Діапазон відстаней (у борах)
distances = np.linspace(0.8, 8.0, 40)

energies_rhf = []
energies_uhf = []
s2_values = []

print("Розрахунок PES для H2 (RHF vs UHF)...")
print("=" * 60)
print(f"{'R (bohr)':<10} {'E_RHF (Ha)':<14} {'E_UHF (Ha)':<14} {'⟨S²⟩':<8}")
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

    # UHF розрахунок
    mf_uhf = scf.UHF(mol)
    mf_uhf.conv_tol = 1e-10
    E_uhf = mf_uhf.kernel()
    energies_uhf.append(E_uhf)

    # Спінове забруднення
    s2 = mf_uhf.spin_square()[0]
    s2_values.append(s2)

    print(f"{R:<10.2f} {E_rhf:<14.8f} {E_uhf:<14.8f} {s2:<8.4f}")

energies_rhf = np.array(energies_rhf)
energies_uhf = np.array(energies_uhf)
s2_values = np.array(s2_values)

print("=" * 60)

# Аналіз результатів
idx_min_rhf = np.argmin(energies_rhf)
idx_min_uhf = np.argmin(energies_uhf)

R_e_rhf = distances[idx_min_rhf]
R_e_uhf = distances[idx_min_uhf]
E_min_rhf = energies_rhf[idx_min_rhf]
E_min_uhf = energies_uhf[idx_min_uhf]

# Дисоціаційні межі
E_2H = 2 * (-0.5)  # 2 атоми H
E_dissoc_rhf = energies_rhf[-1]
E_dissoc_uhf = energies_uhf[-1]

print("\nРЕЗУЛЬТАТИ АНАЛІЗУ:")
print("=" * 60)
print("RHF:")
print(f"  R_e = {R_e_rhf:.3f} bohr = {R_e_rhf * 0.529177:.3f} Å")
print(f"  E(R_e) = {E_min_rhf:.6f} Ha")
print(f"  D_e = {E_2H - E_min_rhf:.6f} Ha = {(E_2H - E_min_rhf) * 27.2114:.3f} eV")
print(f"  E(R→∞) = {E_dissoc_rhf:.6f} Ha")
print(f"  Правильна дисоціація? {'✓' if abs(E_dissoc_rhf - E_2H) < 0.01 else '✗'}")

print("\nUHF:")
print(f"  R_e = {R_e_uhf:.3f} bohr = {R_e_uhf * 0.529177:.3f} Å")
print(f"  E(R_e) = {E_min_uhf:.6f} Ha")
print(f"  D_e = {E_2H - E_min_uhf:.6f} Ha = {(E_2H - E_min_uhf) * 27.2114:.3f} eV")
print(f"  E(R→∞) = {E_dissoc_uhf:.6f} Ha")
print(f"  Правильна дисоціація? {'✓' if abs(E_dissoc_uhf - E_2H) < 0.01 else '✗'}")
print(f"  ⟨S²⟩ при R→∞: {s2_values[-1]:.4f}")

print("\nЕкспериментальні дані:")
print("  R_e = 1.401 bohr = 0.741 Å")
print("  D_e ≈ 4.75 eV")
print("=" * 60)

# Візуалізація
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Графік 1: PES
ax1.plot(distances, energies_rhf, "b-", linewidth=2.5, label="RHF")
ax1.plot(distances, energies_uhf, "r--", linewidth=2.5, label="UHF")
ax1.axhline(
    y=E_2H, color="green", linestyle=":", linewidth=2, label=f"2×E(H) = {E_2H:.3f} Ha"
)
ax1.plot(R_e_rhf, E_min_rhf, "bo", markersize=10, label=f"RHF min: R={R_e_rhf:.2f}")
ax1.plot(R_e_uhf, E_min_uhf, "ro", markersize=10, label=f"UHF min: R={R_e_uhf:.2f}")

ax1.set_ylabel("Енергія E (Ha)", fontsize=13, fontweight="bold")
ax1.set_title(
    "Крива потенційної енергії H$_2$: RHF vs UHF", fontsize=15, fontweight="bold"
)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11, loc="upper right")
ax1.set_ylim([-1.2, -0.4])

# Виділення області розбіжності
ax1.axvspan(
    3.0, 8.0, alpha=0.15, color="red", label="Область некоректної дисоціації RHF"
)
ax1.text(
    5.5,
    -0.5,
    "RHF: H⁺ + H⁻",
    fontsize=11,
    color="blue",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
ax1.text(
    5.5,
    -0.95,
    "UHF: H + H ✓",
    fontsize=11,
    color="red",
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
)

# Графік 2: Спінове забруднення
ax2.plot(
    distances,
    s2_values,
    "purple",
    linewidth=2.5,
    marker="o",
    markersize=4,
    label="⟨S²⟩ (UHF)",
)
ax2.axhline(
    y=0.0,
    color="green",
    linestyle="--",
    linewidth=2,
    label="Очікується: 0.0 (чистий синглет)",
)
ax2.axhline(y=1.0, color="orange", linestyle="--", linewidth=2, label="Триплет: 2.0")

ax2.set_xlabel("Міжядерна відстань R (bohr)", fontsize=13, fontweight="bold")
ax2.set_ylabel("⟨S²⟩", fontsize=13, fontweight="bold")
ax2.set_title("Спінове забруднення в UHF", fontsize=15, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)
ax2.set_ylim([-0.1, 1.2])

# Позначення областей
ax2.axvspan(0.8, 2.5, alpha=0.15, color="green")
ax2.axvspan(2.5, 8.0, alpha=0.15, color="red")
ax2.text(1.5, 0.9, "Чистий синглет", fontsize=10, ha="center")
ax2.text(5.0, 0.9, "Сильне забруднення!", fontsize=10, ha="center", color="red")

plt.tight_layout()
plt.savefig("h2_rhf_vs_uhf.png", dpi=300, bbox_inches="tight")
print("\nГрафіки збережено: h2_rhf_vs_uhf.png")
plt.show()

# Збереження даних
np.savez(
    "h2_pes_comparison.npz",
    distances=distances,
    E_rhf=energies_rhf,
    E_uhf=energies_uhf,
    s2=s2_values,
)
print("Дані збережено: h2_pes_comparison.npz")
