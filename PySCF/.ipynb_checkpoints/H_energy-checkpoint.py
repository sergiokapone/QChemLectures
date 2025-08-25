from pyscf import gto, scf
import matplotlib.pyplot as plt


# Константа для перекладу Hartree → eV
HARTREE_TO_EV = 27.211386245988

# Список базисов для сравнения
basis_set_list = ['STO-3G', '6-31G', '6-31G(d)', 'cc-pVDZ', 'cc-pVTZ', 'cc-pVQZ']


energies_hartree = []
energies_ev = []

print(f"{'Базис':<10} {'E(Hartree)':>12} {'E(eV)':>12}")
print("-"*36)

for basis in basis_set_list:
    mol = gto.M(atom='H 0 0 0',
                charge=0,
                spin=1,      # один електрон → Nα-Nβ=1
                unit='Bohr',
                basis=basis)
    mf = scf.UHF(mol)
    mf.verbose = 0
    energy_hartree = mf.kernel()
    energy_ev = energy_hartree * HARTREE_TO_EV
    energies_hartree.append(energy_hartree)
    energies_ev.append(energy_ev)
    print(f"{basis:<10} {energy_hartree:12.6f} {energy_ev:12.6f}")

# График зависимости энергии от базиса
plt.figure(figsize=(8,5))
plt.plot(basis_set_list, energies_ev, marker='o', linestyle='-', color='blue')
plt.axhline(-13.6057, color='red', linestyle='--', label='Точне значення')
plt.xlabel('Базис')
plt.ylabel('Енергія, eV')
plt.title('Залежність енергії H-атома від базису')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
