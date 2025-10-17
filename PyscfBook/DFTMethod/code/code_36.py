# ============================================================
# code_36.py - Крива потенційної енергії H2
# ============================================================
from pyscf import gto, dft
import numpy as np
import matplotlib.pyplot as plt

# Діапазон відстаней
distances = np.linspace(0.4, 4.0, 40)

# Функціонали для порівняння
functionals = {
    'PBE': 'pbe',
    'B3LYP': 'b3lyp',
    'PBE0': 'pbe0'
}

results = {name: [] for name in functionals}

print("Розрахунок кривої потенційної енергії H2")
print("=" * 60)

for r in distances:
    mol = gto.M(
        atom=f'H 0 0 0; H 0 0 {r}',
        basis='cc-pvqz',
        unit='angstrom'
    )

    for name, xc in functionals.items():
        mf = dft.RKS(mol)
        mf.xc = xc
        mf.verbose = 0
        energy = mf.kernel()
        results[name].append(energy)

    if int(r * 10) % 5 == 0:
        print(f"r = {r:.2f} Å завершено")

# Знаходження мінімумів
print("\n" + "=" * 60)
print("Рівноважні параметри:")
print("=" * 60)
print(f"{'Функціонал':<12} {'r_e (Å)':<12} {'D_e (eV)':<12}")
print("-" * 60)

for name in functionals:
    energies = np.array(results[name])
    min_idx = np.argmin(energies)
    r_e = distances[min_idx]
    D_e = (energies[-1] - energies[min_idx]) * 27.211  # eV

    print(f"{name:<12} {r_e:<12.4f} {D_e:<12.4f}")

print("-" * 60)
print(f"{'Експеримент':<12} {0.741:<12.4f} {4.75:<12.4f}")

# Графік
plt.figure(figsize=(10, 6))
for name in functionals:
    energies = np.array(results[name])
    # Відносно мінімуму в еВ
    E_rel = (energies - np.min(energies)) * 27.211
    plt.plot(distances, E_rel, '-o', label=name, markersize=3)

plt.xlabel('Відстань H-H (Å)', fontsize=12)
plt.ylabel('Енергія відносно мінімуму (eV)', fontsize=12)
plt.title('Крива потенційної енергії H₂', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0.4, 4.0)
plt.ylim(-0.5, 6)
plt.tight_layout()
plt.savefig('h2_pes_dft.png', dpi=300)
print("\nГрафік збережено: h2_pes_dft.png")

