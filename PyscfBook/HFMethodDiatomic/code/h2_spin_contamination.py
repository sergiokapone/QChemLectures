"""
Детальний аналіз спінового забруднення в UHF для H2
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf


def analyze_spin_contamination(R, basis="cc-pvdz"):
    """
    Аналізує спінове забруднення при заданій відстані R
    """
    mol = gto.Mole()
    mol.atom = f"H 0 0 0; H 0 0 {R}"
    mol.basis = basis
    mol.charge = 0
    mol.spin = 0
    mol.unit = "Bohr"
    mol.verbose = 0
    mol.build()

    # UHF розрахунок
    mf = scf.UHF(mol)
    mf.kernel()

    # Спінові характеристики
    s2, ss = mf.spin_square()

    # Матриці густини
    dm_alpha = mf.make_rdm1()[0]
    dm_beta = mf.make_rdm1()[1]

    # Перекриття α та β орбіталей
    overlap = mol.intor("int1e_ovlp")
    mo_alpha = mf.mo_coeff[0]
    mo_beta = mf.mo_coeff[1]

    # Обчислення перекриття між α та β HOMO
    n_occ = mol.nelec[0]
    homo_alpha = mo_alpha[:, n_occ - 1]
    homo_beta = mo_beta[:, n_occ - 1]

    overlap_homo = abs(homo_alpha @ overlap @ homo_beta)

    return {
        "E": mf.e_tot,
        "s2": s2,
        "s": ss,
        "overlap_homo": overlap_homo,
        "dm_alpha": dm_alpha,
        "dm_beta": dm_beta,
        "mo_energy_alpha": mf.mo_energy[0][:5],
        "mo_energy_beta": mf.mo_energy[1][:5],
    }


# Детальний аналіз при різних відстанях
distances = [1.4, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]

print("\n" + "=" * 80)
print("ДЕТАЛЬНИЙ АНАЛІЗ СПІНОВОГО ЗАБРУДНЕННЯ В H2 (UHF)")
print("=" * 80)
print(
    f"{'R (bohr)':<10} {'E (Ha)':<14} {'⟨S²⟩':<10} {'S':<10} "
    f"{'Δ⟨S²⟩':<12} {'Overlap':<10}"
)
print("-" * 80)

results = []
for R in distances:
    res = analyze_spin_contamination(R)
    delta_s2 = res["s2"] - 0.0  # Для синглету S(S+1) = 0

    results.append(res)

    print(
        f"{R:<10.2f} {res['E']:<14.8f} {res['s2']:<10.6f} {res['s']:<10.4f} "
        f"{delta_s2:<12.6f} {res['overlap_homo']:<10.6f}"
    )

print("=" * 80)
print("\nІНТЕРПРЕТАЦІЯ:")
print("-" * 80)
print("⟨S²⟩ = 0.00: чистий синглет (правильний стан)")
print("⟨S²⟩ ≈ 1.00: сильне забруднення триплетом")
print("Overlap ≈ 1.00: α та β орбіталі ідентичні (→ RHF)")
print("Overlap ≈ 0.00: α та β орбіталі різні (сильна поляризація)")
print("=" * 80)

# Аналіз при R = 5.0 bohr (сильне забруднення)
print("\n" + "=" * 80)
print("ДЕТАЛЬНИЙ АНАЛІЗ ПРИ R = 5.0 BOHR")
print("=" * 80)

res_5 = analyze_spin_contamination(5.0)

print("\nОрбітальні енергії (перші 5 МО):")
print("-" * 80)
print("α-спін:")
for i, e in enumerate(res_5["mo_energy_alpha"]):
    print(f"  МО {i + 1}: ε = {e:10.6f} Ha")

print("\nβ-спін:")
for i, e in enumerate(res_5["mo_energy_beta"]):
    print(f"  МО {i + 1}: ε = {e:10.6f} Ha")

print(
    f"\nРізниця HOMO(α) - HOMO(β): {res_5['mo_energy_alpha'][0] - res_5['mo_energy_beta'][0]:.6f} Ha"
)
print("(Для чистого RHF ця різниця = 0)")

# Візуалізація
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. ⟨S²⟩ vs R
ax1 = fig.add_subplot(gs[0, :])
R_fine = np.linspace(1.0, 8.0, 50)
s2_fine = []
for r in R_fine:
    res = analyze_spin_contamination(r)
    s2_fine.append(res["s2"])

ax1.plot(R_fine, s2_fine, "b-", linewidth=3, label="UHF ⟨S²⟩")
ax1.axhline(0.0, color="green", linestyle="--", linewidth=2, label="Синглет (S=0)")
ax1.axhline(2.0, color="red", linestyle="--", linewidth=2, label="Триплет (S=1)")
ax1.fill_between(R_fine, 0, s2_fine, alpha=0.3, color="orange")

for R in distances:
    idx = np.argmin(abs(R_fine - R))
    ax1.plot(R, s2_fine[idx], "ro", markersize=8)
    ax1.text(R, s2_fine[idx] + 0.1, f"{s2_fine[idx]:.2f}", ha="center", fontsize=9)

ax1.set_xlabel("R (bohr)", fontsize=13, fontweight="bold")
ax1.set_ylabel("⟨S²⟩", fontsize=13, fontweight="bold")
ax1.set_title("Спінове забруднення як функція відстані", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
ax1.set_ylim([-0.2, 2.5])

# 2. Перекриття HOMO vs R
ax2 = fig.add_subplot(gs[1, 0])
overlaps = [analyze_spin_contamination(r)["overlap_homo"] for r in R_fine]
ax2.plot(R_fine, overlaps, "purple", linewidth=3)
ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
ax2.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
ax2.set_xlabel("R (bohr)", fontsize=12)
ax2.set_ylabel("|⟨HOMO α|HOMO β⟩|", fontsize=12)
ax2.set_title("Перекриття α/β орбіталей", fontsize=13, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.set_ylim([-0.1, 1.1])

# 3. Енергія vs R
ax3 = fig.add_subplot(gs[1, 1])
energies = [analyze_spin_contamination(r)["E"] for r in R_fine]
ax3.plot(R_fine, energies, "darkred", linewidth=3)
ax3.axhline(-1.0, color="green", linestyle="--", linewidth=2, label="2×E(H)")
ax3.set_xlabel("R (bohr)", fontsize=12)
ax3.set_ylabel("E (Ha)", fontsize=12)
ax3.set_title("Енергія (UHF)", fontsize=13, fontweight="bold")
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. Таблиця значень
ax4 = fig.add_subplot(gs[2, :])
ax4.axis("off")

table_data = []
for i, R in enumerate(distances):
    res = results[i]
    table_data.append(
        [
            f"{R:.1f}",
            f"{res['E']:.5f}",
            f"{res['s2']:.4f}",
            f"{res['s2']:.4f}",
            f"{res['overlap_homo']:.4f}",
        ]
    )

table = ax4.table(
    cellText=table_data,
    colLabels=["R (bohr)", "E (Ha)", "⟨S²⟩", "Δ⟨S²⟩", "Overlap"],
    cellLoc="center",
    loc="center",
    bbox=[0.1, 0.0, 0.8, 0.9],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Стилізація
for i in range(len(distances) + 1):
    for j in range(5):
        cell = table[(i, j)]
        if i == 0:
            cell.set_facecolor("#4CAF50")
            cell.set_text_props(weight="bold", color="white")
        else:
            # Кольорове кодування ⟨S²⟩
            s2_val = float(table_data[i - 1][2])
            if s2_val < 0.1:
                cell.set_facecolor("#E8F5E9")  # Зелений
            elif s2_val < 0.5:
                cell.set_facecolor("#FFF9C4")  # Жовтий
            else:
                cell.set_facecolor("#FFCDD2")  # Червоний

plt.suptitle(
    "Повний аналіз спінового забруднення в H$_2$",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)
plt.savefig("h2_spin_contamination_detailed.png", dpi=300, bbox_inches="tight")
print("\nГрафіки збережено: h2_spin_contamination_detailed.png")
plt.show()

# Фізичне пояснення
print("\n" + "=" * 80)
print("ФІЗИЧНЕ ПОЯСНЕННЯ СПІНОВОГО ЗАБРУДНЕННЯ:")
print("=" * 80)
print("""
При малих R (навколо рівноваги):
  • Електрони сильно зв'язані
  • α та β орбіталі практично ідентичні
  • UHF ≈ RHF, ⟨S²⟩ ≈ 0

При великих R (дисоціація):
  • Кожен електрон локалізується на своєму атомі
  • α-орбіталь → 1s_A, β-орбіталь → 1s_B
  • Орбіталі різні (overlap → 0)
  • UHF хвильова функція стає сумішшю синглету та триплету
  • ⟨S²⟩ → 1.0 (формально триплет, але це артефакт методу!)

Правильний опис вимагає багатоконфігураційних методів (CASSCF, MRCI).
""")
print("=" * 80)
