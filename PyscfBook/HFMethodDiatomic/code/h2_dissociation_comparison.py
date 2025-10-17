"""
Повне порівняння RHF, UHF та точного розв'язку (FCI) для H2
Демонстрація проблеми дисоціації
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci


def calculate_h2_curve(distances, method="rhf", basis="cc-pvdz"):
    """
    Обчислює PES для H2 заданим методом

    method: 'rhf', 'uhf', 'fci'
    """
    energies = []
    s2_values = []

    for R in distances:
        mol = gto.Mole()
        mol.atom = f"H 0 0 0; H 0 0 {R}"
        mol.basis = basis
        mol.charge = 0
        mol.spin = 0
        mol.unit = "Bohr"
        mol.verbose = 0
        mol.build()

        if method.lower() == "rhf":
            mf = scf.RHF(mol)
            E = mf.kernel()
            s2 = 0.0

        elif method.lower() == "uhf":
            mf = scf.UHF(mol)
            E = mf.kernel()
            s2, _ = mf.spin_square()

        elif method.lower() == "fci":
            # Спочатку RHF як початкове наближення
            mf = scf.RHF(mol)
            mf.kernel()
            # Потім Full CI
            cisolver = fci.FCI(mf)
            E, civec = cisolver.kernel()
            s2 = 0.0  # FCI дає чистий синглет

        energies.append(E)
        s2_values.append(s2)

    return np.array(energies), np.array(s2_values)


# Діапазон відстаней
distances = np.linspace(0.8, 8.0, 35)

print("\n" + "=" * 70)
print("ПОРІВНЯННЯ МЕТОДІВ ДЛЯ H2: RHF vs UHF vs FCI")
print("=" * 70)
print("Обчислення (це може зайняти кілька хвилин)...")

# Розрахунки
E_rhf, _ = calculate_h2_curve(distances, method="rhf")
E_uhf, s2_uhf = calculate_h2_curve(distances, method="uhf")
E_fci, _ = calculate_h2_curve(distances, method="fci")

# Аналіз
E_2H = -1.0  # Точна енергія 2×H

print("\nРЕЗУЛЬТАТИ:")
print("=" * 70)

# Знаходження мінімумів
idx_rhf = np.argmin(E_rhf)
idx_uhf = np.argmin(E_uhf)
idx_fci = np.argmin(E_fci)

methods_data = {
    "RHF": {
        "R_e": distances[idx_rhf],
        "E_min": E_rhf[idx_rhf],
        "E_dissoc": E_rhf[-1],
        "D_e": E_2H - E_rhf[idx_rhf],
    },
    "UHF": {
        "R_e": distances[idx_uhf],
        "E_min": E_uhf[idx_uhf],
        "E_dissoc": E_uhf[-1],
        "D_e": E_2H - E_uhf[idx_uhf],
    },
    "FCI": {
        "R_e": distances[idx_fci],
        "E_min": E_fci[idx_fci],
        "E_dissoc": E_fci[-1],
        "D_e": E_2H - E_fci[idx_fci],
    },
}

for method, data in methods_data.items():
    print(f"\n{method}:")
    print(f"  R_e = {data['R_e']:.3f} bohr = {data['R_e'] * 0.529177:.3f} Å")
    print(f"  E(R_e) = {data['E_min']:.6f} Ha")
    print(f"  D_e = {data['D_e']:.6f} Ha = {data['D_e'] * 27.2114:.3f} eV")
    print(f"  E(R→∞) = {data['E_dissoc']:.6f} Ha")
    print(f"  Δ від 2×E(H): {abs(data['E_dissoc'] - E_2H) * 1000:.2f} mHa")

print(f"\nЕкспериментальні дані:")
print(f"  R_e = 1.401 bohr = 0.741 Å")
print(f"  D_e = 4.75 eV = 0.1745 Ha")

# Кореляційна енергія
print("\n" + "-" * 70)
print("КОРЕЛЯЦІЙНА ЕНЕРГІЯ E_corr = E_FCI - E_HF:")
print("-" * 70)
print(f"{'R (bohr)':<10} {'E_corr (mHa)':<15} {'% від D_e':<12}")
print("-" * 70)

selected_R = [1.4, 2.0, 3.0, 4.0, 6.0, 8.0]
for R in selected_R:
    idx = np.argmin(abs(distances - R))
    E_corr = (E_fci[idx] - E_rhf[idx]) * 1000  # mHa
    percent = abs(E_corr) / (methods_data["FCI"]["D_e"] * 1000) * 100
    print(f"{R:<10.1f} {E_corr:<15.2f} {percent:<12.1f}%")

print("=" * 70)

# Побудова графіків
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# 1. Повні PES
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(distances, E_rhf, "b-", linewidth=3, label="RHF", alpha=0.8)
ax1.plot(distances, E_uhf, "r--", linewidth=3, label="UHF", alpha=0.8)
ax1.plot(distances, E_fci, "g-.", linewidth=3, label="FCI (точний)", alpha=0.8)
ax1.axhline(
    E_2H,
    color="black",
    linestyle=":",
    linewidth=2,
    label=f"2×E(H) = {E_2H:.3f} Ha",
    alpha=0.6,
)

# Позначення мінімумів
ax1.plot(distances[idx_rhf], E_rhf[idx_rhf], "bo", markersize=10)
ax1.plot(distances[idx_uhf], E_uhf[idx_uhf], "ro", markersize=10)
ax1.plot(distances[idx_fci], E_fci[idx_fci], "go", markersize=10)

ax1.set_xlabel("Міжядерна відстань R (bohr)", fontsize=13, fontweight="bold")
ax1.set_ylabel("Енергія E (Ha)", fontsize=13, fontweight="bold")
ax1.set_title("Криві потенційної енергії H$_2$", fontsize=15, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=12, loc="upper right")
ax1.set_ylim([-1.25, -0.3])

# Виділення областей
ax1.axvspan(0.8, 2.5, alpha=0.1, color="green", label="Рівновага")
ax1.axvspan(3.5, 8.0, alpha=0.1, color="red", label="Дисоціація")

# 2. Zoom навколо мінімуму
ax2 = fig.add_subplot(gs[1, 0])
mask = (distances >= 0.8) & (distances <= 2.5)
ax2.plot(distances[mask], E_rhf[mask], "b-", linewidth=2.5, label="RHF")
ax2.plot(distances[mask], E_uhf[mask], "r--", linewidth=2.5, label="UHF")
ax2.plot(distances[mask], E_fci[mask], "g-.", linewidth=2.5, label="FCI")

ax2.set_xlabel("R (bohr)", fontsize=12)
ax2.set_ylabel("E (Ha)", fontsize=12)
ax2.set_title("Zoom: область рівноваги", fontsize=13, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# 3. Zoom дисоціація
ax3 = fig.add_subplot(gs[1, 1])
mask = (distances >= 3.5) & (distances <= 8.0)
ax3.plot(distances[mask], E_rhf[mask], "b-", linewidth=2.5, label="RHF")
ax3.plot(distances[mask], E_uhf[mask], "r--", linewidth=2.5, label="UHF")
ax3.plot(distances[mask], E_fci[mask], "g-.", linewidth=2.5, label="FCI")
ax3.axhline(E_2H, color="black", linestyle=":", linewidth=2, alpha=0.6)

ax3.set_xlabel("R (bohr)", fontsize=12)
ax3.set_ylabel("E (Ha)", fontsize=12)
ax3.set_title("Zoom: дисоціація", fontsize=13, fontweight="bold")
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

# Анотації
ax3.annotate(
    "RHF → H⁺ + H⁻\n(неправильно!)",
    xy=(7, E_rhf[-1]),
    xytext=(6, -0.8),
    arrowprops=dict(arrowstyle="->", color="blue", lw=2),
    fontsize=11,
    color="blue",
    fontweight="bold",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
)

ax3.annotate(
    "UHF, FCI → 2H\n(правильно)",
    xy=(7, E_uhf[-1]),
    xytext=(5.5, -1.05),
    arrowprops=dict(arrowstyle="->", color="green", lw=2),
    fontsize=11,
    color="green",
    fontweight="bold",
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
)

# 4. Кореляційна енергія
ax4 = fig.add_subplot(gs[2, 0])
E_corr_rhf = (E_fci - E_rhf) * 1000  # mHa
E_corr_uhf = (E_fci - E_uhf) * 1000

ax4.plot(distances, E_corr_rhf, "b-", linewidth=2.5, label="E$_{corr}$ (FCI - RHF)")
ax4.plot(distances, E_corr_uhf, "r--", linewidth=2.5, label="E$_{corr}$ (FCI - UHF)")
ax4.axhline(0, color="black", linestyle=":", linewidth=1, alpha=0.5)

ax4.set_xlabel("R (bohr)", fontsize=12)
ax4.set_ylabel("Кореляційна енергія (mHa)", fontsize=12)
ax4.set_title("Кореляційна енергія", fontsize=13, fontweight="bold")
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)

# 5. Спінове забруднення
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(distances, s2_uhf, "purple", linewidth=2.5, marker="o", markersize=4)
ax5.axhline(0.0, color="green", linestyle="--", linewidth=2, label="Синглет")
ax5.axhline(2.0, color="red", linestyle="--", linewidth=2, label="Триплет")
ax5.fill_between(distances, 0, s2_uhf, alpha=0.3, color="orange")

ax5.set_xlabel("R (bohr)", fontsize=12)
ax5.set_ylabel("⟨S²⟩ (UHF)", fontsize=12)
ax5.set_title("Спінове забруднення", fontsize=13, fontweight="bold")
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=10)
ax5.set_ylim([-0.2, 2.3])

plt.suptitle(
    "Повне порівняння методів для дисоціації H$_2$",
    fontsize=17,
    fontweight="bold",
    y=0.995,
)

plt.savefig("h2_complete_comparison.png", dpi=300, bbox_inches="tight")
print("\nГрафіки збережено: h2_complete_comparison.png")
plt.show()

# Збереження даних
np.savez(
    "h2_methods_comparison.npz",
    distances=distances,
    E_rhf=E_rhf,
    E_uhf=E_uhf,
    E_fci=E_fci,
    s2_uhf=s2_uhf,
)
print("Дані збережено: h2_methods_comparison.npz")

# Підсумкова таблиця
print("\n" + "=" * 70)
print("ПІДСУМКОВА ТАБЛИЦЯ (для LaTeX):")
print("=" * 70)
print(r"\begin{tabular}{lcccc}")
print(r"\toprule")
print(r"Метод & $R_e$ (bohr) & $E(R_e)$ (Ha) & $D_e$ (eV) & $E(R\to\infty)$ (Ha) \\")
print(r"\midrule")
for method, data in methods_data.items():
    print(
        f"{method} & {data['R_e']:.3f} & {data['E_min']:.4f} & "
        f"{data['D_e'] * 27.2114:.2f} & {data['E_dissoc']:.4f} \\\\"
    )
print(r"\midrule")
print(r"Експеримент & 1.401 & --- & 4.75 & $-1.000$ \\")
print(r"\bottomrule")
print(r"\end{tabular}")
print("=" * 70)
