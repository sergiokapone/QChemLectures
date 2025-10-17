# ============================================================
# code_47.py - Проблема дисоціації (RKS vs UKS)
# ============================================================
from pyscf import gto, dft
import numpy as np
import matplotlib.pyplot as plt

print("Дисоціація H2: RKS vs UKS")
print("=" * 80)

distances = np.linspace(0.5, 6.0, 40)

results_diss = {
    'RKS-PBE': [],
    'UKS-PBE': [],
    'RKS-PBE0': [],
    'UKS-PBE0': [],
}

print("\nРозрахунок кривих дисоціації...")

for r in distances:
    mol = gto.M(
        atom=f'H 0 0 0; H 0 0 {r}',
        basis='cc-pvqz',
        unit='angstrom'
    )

    # RKS PBE
    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.verbose = 0
    results_diss['RKS-PBE'].append(mf.kernel())

    # UKS PBE
    mf = dft.UKS(mol)
    mf.xc = 'pbe'
    mf.verbose = 0
    results_diss['UKS-PBE'].append(mf.kernel())

    # RKS PBE0
    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.verbose = 0
    results_diss['RKS-PBE0'].append(mf.kernel())

    # UKS PBE0
    mf = dft.UKS(mol)
    mf.xc = 'pbe0'
    mf.verbose = 0
    results_diss['UKS-PBE0'].append(mf.kernel())

    if int(r * 10) % 10 == 0:
        print(f"  r = {r:.1f} Å")

# Аналіз результатів
print("\n" + "=" * 80)
print("Асимптотична поведінка (r → ∞):")
print("-" * 80)

for name, energies in results_diss.items():
    E_inf = energies[-1]
    E_min = min(energies)
    D_e = (E_inf - E_min) * 27.211
    print(f"{name:<15} E(∞) = {E_inf:.8f} Ha,  D_e = {D_e:.4f} eV")

# Порівняння з точною межею (2 × E(H))
mol_H = gto.M(atom='H 0 0 0', basis='cc-pvqz')
mf_H = dft.RKS(mol_H)
mf_H.xc = 'pbe'
mf_H.verbose = 0
E_H_pbe = mf_H.kernel()

mf_H.xc = 'pbe0'
E_H_pbe0 = mf_H.kernel()

print("\n" + "=" * 80)
print("Порівняння з E(∞) = 2×E(H atom):")
print("-" * 80)
print(f"2×E(H) PBE  = {2*E_H_pbe:.8f} Ha")
print(f"2×E(H) PBE0 = {2*E_H_pbe0:.8f} Ha")

print(f"\nВідхилення RKS-PBE:  {(results_diss['RKS-PBE'][-1] - 2*E_H_pbe)*27.211*1000:.2f} meV")
print(f"Відхилення UKS-PBE:  {(results_diss['UKS-PBE'][-1] - 2*E_H_pbe)*27.211*1000:.2f} meV")
print(f"Відхилення RKS-PBE0: {(results_diss['RKS-PBE0'][-1] - 2*E_H_pbe0)*27.211*1000:.2f} meV")
print(f"Відхилення UKS-PBE0: {(results_diss['UKS-PBE0'][-1] - 2*E_H_pbe0)*27.211*1000:.2f} meV")

# Графік
plt.figure(figsize=(12, 6))

for name, energies in results_diss.items():
    E_rel = (np.array(energies) - min(energies)) * 27.211
    linestyle = '-' if 'UKS' in name else '--'
    plt.plot(distances, E_rel, linestyle, label=name, linewidth=2)

plt.axhline(y=4.75, color='k', linestyle=':', label='Експеримент (4.75 eV)')
plt.xlabel('Відстань H-H (Å)', fontsize=12)
plt.ylabel('Енергія відносно мінімуму (eV)', fontsize=12)
plt.title('Дисоціація H₂: RKS vs UKS', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0.5, 6.0)
plt.ylim(-0.5, 6)
plt.tight_layout()
plt.savefig('h2_dissociation_rks_uks.png', dpi=300)

print("\nГрафік збережено: h2_dissociation_rks_uks.png")
print("\nВисновок:")
print("- UKS правильно описує дисоціацію (E → 2×E(H))")
print("- RKS має помилку через обмеження єдиної детермінанти")
print("- Для гібридів (PBE0) помилка менша через HF обмін")

