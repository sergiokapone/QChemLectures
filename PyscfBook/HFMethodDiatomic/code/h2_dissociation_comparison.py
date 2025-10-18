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
print("ПОРІВНЯННЯ МЕТОДІВ ДЛЯ H2: RHF vs FCI")
print("=" * 70)
print("Обчислення (це може зайняти кілька хвилин)...")

# Розрахунки
E_rhf, _ = calculate_h2_curve(distances, method="rhf")
E_fci, _ = calculate_h2_curve(distances, method="fci")

# Аналіз
E_2H = -1.0  # Точна енергія 2×H

print("\nРЕЗУЛЬТАТИ:")
print("=" * 70)

# Знаходження мінімумів
idx_rhf = np.argmin(E_rhf)
idx_fci = np.argmin(E_fci)

methods_data = {
    "RHF": {
        "R_e": distances[idx_rhf],
        "E_min": E_rhf[idx_rhf],
        "E_dissoc": E_rhf[-1],
        "D_e": E_2H - E_rhf[idx_rhf],
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

selected_R = np.linspace(1.2, 8, 10)
for R in selected_R:
    idx = np.argmin(abs(distances - R))
    E_corr = (E_fci[idx] - E_rhf[idx]) * 1000  # mHa
    percent = abs(E_corr) / (methods_data["FCI"]["D_e"] * 1000) * 100
    print(f"{R:<10.1f} {E_corr:<15.2f} {percent:<12.1f}%")

print("=" * 70)


fig = plt.figure(figsize=(16, 15), constrained_layout=False)
# ТРИ ряда, ДВА стовпці — всі рядки рівні за висотою
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.45, wspace=0.35)

# 1) Повна PES (ряд 0, обидва стовпці)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(distances, E_rhf, "-", color="tab:blue", lw=2.8, label="RHF", alpha=0.9)
ax1.plot(distances, E_fci, "-.", color="tab:green", lw=2.8, label="FCI (точний)", alpha=0.9)
ax1.axhline(E_2H, color="k", ls=":", lw=1.6, alpha=0.7)
ax1.plot(distances[idx_rhf], E_rhf[idx_rhf], "o", color="tab:blue", ms=8)
ax1.plot(distances[idx_fci], E_fci[idx_fci], "o", color="tab:green", ms=8)
ax1.set_ylim([-1.25, -0.3])
ax1.set_title("Криві потенційної енергії H$_2$", fontsize=16, fontweight="bold")
ax1.set_xlabel("Міжядерна відстань $R$ (bohr)", fontsize=13)
ax1.set_ylabel("Енергія $E$ (Ha)", fontsize=13)
ax1.grid(alpha=0.25)
ax1.legend(fontsize=11, loc="upper right")
ax1.axvspan(0.8, 2.5, alpha=0.08, color="green")
ax1.axvspan(3.5, 8.0, alpha=0.08, color="red")

# 2) Zoom рівновага (ряд 1, стовпець 0)
ax2 = fig.add_subplot(gs[1, 0])
mask_eq = (distances >= 0.8) & (distances <= 2.5)
ax2.plot(distances[mask_eq], E_rhf[mask_eq], "-", color="tab:blue", lw=2.5, label="RHF")
ax2.plot(distances[mask_eq], E_fci[mask_eq], "-.", color="tab:green", lw=2.5, label="FCI")
ax2.set_title("Zoom: область рівноваги", fontsize=14, fontweight="bold")
ax2.set_xlabel("R (bohr)")
ax2.set_ylabel("E (Ha)")
ax2.grid(alpha=0.25)
ax2.legend(fontsize=10)

# 3) Zoom дисоціація (ряд 1, стовпець 1)
ax3 = fig.add_subplot(gs[1, 1])
mask_dis = (distances >= 3.5) & (distances <= 8.0)
ax3.plot(distances[mask_dis], E_rhf[mask_dis], "-", color="tab:blue", lw=2.5, label="RHF")
ax3.plot(distances[mask_dis], E_fci[mask_dis], "-.", color="tab:green", lw=2.5, label="FCI")
ax3.axhline(E_2H, color="k", ls=":", lw=1.4, alpha=0.7)
ax3.set_title("Zoom: дисоціація", fontsize=14, fontweight="bold")
ax3.set_xlabel("R (bohr)")
ax3.set_ylabel("E (Ha)")
ax3.grid(alpha=0.25)
ax3.legend(fontsize=10)

ax3.annotate(
    "RHF → H⁺ + H⁻\n(неправильно!)",
    xy=(7, E_rhf[-1]), xytext=(6, -0.8),
    arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.8),
    fontsize=11, color="tab:blue", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="wheat", alpha=0.8),
)
ax3.annotate(
    "FCI → 2H\n(правильно)",
    xy=(7, E_fci[-1]), xytext=(5.4, -1.05),
    arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.8),
    fontsize=11, color="tab:green", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="lightgreen", alpha=0.8),
)

# 4) Кореляційна енергія (ряд 2, обидва стовпці)
ax4 = fig.add_subplot(gs[2, :])
E_corr_rhf = (E_fci - E_rhf) * 1000.0  # mHa
ax4.plot(distances, E_corr_rhf, "-", color="tab:blue", lw=2.8, label="E$_{corr}$ (FCI − RHF)")
ax4.axhline(0, color="k", ls=":", lw=1.0, alpha=0.6)
ax4.set_title("Кореляційна енергія", fontsize=14, fontweight="bold")
ax4.set_xlabel("R (bohr)")
ax4.set_ylabel("Кореляційна енергія (mHa)")
ax4.grid(alpha=0.25)
ax4.legend(fontsize=10)

# Заголовок та фіналізація
plt.suptitle("Порівняння RHF та FCI для дисоціації H$_2$", fontsize=18, fontweight="bold", y=0.995)
# plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("h2_RHF_vs_FCI.pdf", dpi=300, bbox_inches="tight")
print("Графік збережено як: h2_RHF_vs_FCI.pdf")
plt.show()

# Збереження даних
np.savez(
    "h2_methods_comparison.npz",
    distances=distances,
    E_rhf=E_rhf,
    E_fci=E_fci,
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

