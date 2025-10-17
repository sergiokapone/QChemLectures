"""
Систематичне дослідження впливу базисного набору на H2+
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf
from pyscf.geomopt.geometric_solver import optimize

# Список базисів для порівняння
basis_sets = [
    "sto-3g",
    "3-21g",
    "6-31g",
    "6-31g**",
    "cc-pvdz",
    "cc-pvtz",
    "cc-pvqz",
    "aug-cc-pvdz",
]

# Точні значення (експериментальні / high-level теорія)
R_exact = 2.00  # bohr
E_exact = -0.5689  # Ha

results = {"basis": [], "n_basis": [], "R_e": [], "E_e": [], "D_e": [], "time": []}

print("\n" + "=" * 70)
print("ЗБІЖНІСТЬ ПО БАЗИСНОМУ НАБОРУ ДЛЯ H2+")
print("=" * 70)
print(
    f"{'Базис':<15} {'N_AO':<6} {'R_e (bohr)':<12} {'E_e (Ha)':<14} "
    f"{'D_e (eV)':<10} {'Похибка (mHa)':<15}"
)
print("-" * 70)


for basis in basis_sets:
    t_start = time.time()

    try:
        # Створення молекули
        mol = gto.Mole()
        mol.atom = """
            H  0.0  0.0  0.0
            H  0.0  0.0  2.0
        """  # Стартова геометрія
        mol.basis = basis
        mol.charge = 1
        mol.spin = 1
        mol.unit = "Bohr"
        mol.verbose = 0
        mol.build()

        n_ao = mol.nao

        # UHF розрахунок
        mf = scf.UHF(mol)

        # Оптимізація геометрії
        mol_opt = optimize(mf, maxsteps=30)

        # Результати
        coords = mol_opt.atom_coords()
        R_e = np.linalg.norm(coords[1] - coords[0])
        E_e = mf.e_tot
        D_e = -0.5 - E_e  # E(H) = -0.5 Ha

        error = (E_e - E_exact) * 1000  # mHa

        results["basis"].append(basis)
        results["n_basis"].append(n_ao)
        results["R_e"].append(R_e)
        results["E_e"].append(E_e)
        results["D_e"].append(D_e * 27.211)  # eV
        results["time"].append(time.time() - t_start)

        print(
            f"{basis:<15} {n_ao:<6} {R_e:<12.4f} {E_e:<14.6f} "
            f"{D_e * 27.211:<10.3f} {error:<15.2f}"
        )

    except Exception as e:
        print(f"{basis:<15} FAILED: {str(e)[:40]}")
        continue

print("-" * 70)
print(
    f"{'Точне значення':<15} {'---':<6} {R_exact:<12.2f} "
    f"{E_exact:<14.4f} {'2.79':<10} {'---':<15}"
)
print("=" * 70)

# Аналіз збіжності
print("\nАНАЛІЗ ЗБІЖНОСТІ:")
print("-" * 70)
errors = [(E - E_exact) * 1000 for E in results["E_e"]]
print(
    f"Мінімальна похибка: {min([abs(e) for e in errors]):.3f} mHa ({results['basis'][np.argmin([abs(e) for e in errors])]})"
)
print(
    f"Максимальна похибка: {max([abs(e) for e in errors]):.3f} mHa ({results['basis'][np.argmax([abs(e) for e in errors])]})"
)
print(f"\nСередній час розрахунку: {np.mean(results['time']):.2f} с")

# Візуалізація збіжності
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 1. Енергія vs розмір базису
ax1.plot(results["n_basis"], results["E_e"], "bo-", linewidth=2, markersize=8)
ax1.axhline(E_exact, color="r", linestyle="--", linewidth=2, label="Точне значення")
ax1.set_xlabel("Кількість базисних функцій", fontsize=11)
ax1.set_ylabel("Енергія E (Ha)", fontsize=11)
ax1.set_title("Збіжність енергії", fontsize=13, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Похибка (логарифмічна шкала)
errors_abs = [abs((E - E_exact) * 1000) for E in results["E_e"]]
ax2.semilogy(results["n_basis"], errors_abs, "ro-", linewidth=2, markersize=8)
ax2.set_xlabel("Кількість базисних функцій", fontsize=11)
ax2.set_ylabel("Абсолютна похибка (mHa)", fontsize=11)
ax2.set_title("Похибка енергії (log scale)", fontsize=13, fontweight="bold")
ax2.grid(True, alpha=0.3, which="both")

# 3. Рівноважна відстань
ax3.plot(results["n_basis"], results["R_e"], "go-", linewidth=2, markersize=8)
ax3.axhline(R_exact, color="r", linestyle="--", linewidth=2, label="Точне значення")
ax3.set_xlabel("Кількість базисних функцій", fontsize=11)
ax3.set_ylabel("R$_e$ (bohr)", fontsize=11)
ax3.set_title("Збіжність рівноважної відстані", fontsize=13, fontweight="bold")
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. Енергія дисоціації
ax4.plot(results["n_basis"], results["D_e"], "mo-", linewidth=2, markersize=8)
ax4.axhline(2.79, color="r", linestyle="--", linewidth=2, label="Експеримент (2.79 eV)")
ax4.set_xlabel("Кількість базисних функцій", fontsize=11)
ax4.set_ylabel("D$_e$ (eV)", fontsize=11)
ax4.set_title("Енергія дисоціації", fontsize=13, fontweight="bold")
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig("h2plus_basis_convergence.png", dpi=300, bbox_inches="tight")
print("\nГрафіки збереженo: h2plus_basis_convergence.png")
plt.show()

# Таблиця для LaTeX
print("\n" + "=" * 70)
print("ТАБЛИЦЯ ДЛЯ LATEX:")
print("=" * 70)
print(r"\begin{tabular}{lccccc}")
print(r"\toprule")
print(r"Базис & $N_{AO}$ & $R_e$ (bohr) & $E_e$ (Ha) & $D_e$ (eV) & Похибка (mHa) \\")
print(r"\midrule")
for i, basis in enumerate(results["basis"]):
    error = (results["E_e"][i] - E_exact) * 1000
    print(
        f"{basis} & {results['n_basis'][i]} & {results['R_e'][i]:.3f} & "
        f"{results['E_e'][i]:.4f} & {results['D_e'][i]:.2f} & {error:.2f} \\\\"
    )
print(r"\midrule")
print(f"Точне & --- & {R_exact:.2f} & {E_exact:.4f} & 2.79 & --- \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print("=" * 70)

# Збереження результатів
np.savez(
    "h2plus_basis_convergence.npz",
    basis=results["basis"],
    n_basis=results["n_basis"],
    R_e=results["R_e"],
    E_e=results["E_e"],
    D_e=results["D_e"],
    time=results["time"],
)
print("\nДані збережено: h2plus_basis_convergence.npz")
