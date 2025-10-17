"""
Детальний аналіз молекулярних орбіталей H2+
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, tools

# Молекула при рівноважній відстані
mol = gto.Mole()
mol.atom = """
    H  0.0  0.0  -1.0
    H  0.0  0.0   1.0
"""
mol.basis = "cc-pvdz"
mol.charge = 1
mol.spin = 1
mol.unit = "Bohr"
mol.build()

# UHF розрахунок
mf = scf.UHF(mol)
mf.kernel()

print("\n" + "=" * 60)
print("АНАЛІЗ МОЛЕКУЛЯРНИХ ОРБІТАЛЕЙ H2+")
print("=" * 60)

# Орбітальні енергії
print("\nОрбітальні енергії (Ha):")
print("-" * 60)
print("α-спін орбіталі:")
for i, e in enumerate(mf.mo_energy[0][:5]):  # Перші 5
    occ = "зайнята" if i < mol.nelec[0] else "віртуальна"
    print(f"  МО {i + 1}: ε = {e:8.4f} Ha = {e * 27.211:8.2f} eV  ({occ})")

print("\nβ-спін орбіталі:")
for i, e in enumerate(mf.mo_energy[1][:5]):
    occ = "зайнята" if i < mol.nelec[1] else "віртуальна"
    print(f"  МО {i + 1}: ε = {e:8.4f} Ha = {e * 27.211:8.2f} eV  ({occ})")

# HOMO-LUMO gap
homo_alpha = mf.mo_energy[0][mol.nelec[0] - 1]
lumo_alpha = mf.mo_energy[0][mol.nelec[0]]
gap = lumo_alpha - homo_alpha

print("\n" + "-" * 60)
print(f"HOMO (α): ε = {homo_alpha:.4f} Ha = {homo_alpha * 27.211:.2f} eV")
print(f"LUMO (α): ε = {lumo_alpha:.4f} Ha = {lumo_alpha * 27.211:.2f} eV")
print(f"HOMO-LUMO gap: {gap:.4f} Ha = {gap * 27.211:.2f} eV")

# Аналіз коефіцієнтів МО (для α-спіну)
print("\n" + "=" * 60)
print("КОЕФІЦІЄНТИ МОЛЕКУЛЯРНИХ ОРБІТАЛЕЙ (α-спін)")
print("=" * 60)
print(f"Базис: {mol.basis} → {mol.nao} базисних функцій")

# HOMO (зайнята орбіталь)
homo_coeff = mf.mo_coeff[0][:, mol.nelec[0] - 1]
print("\nHOMO (σ_g зв'язуюча):")
for i, c in enumerate(homo_coeff):
    if abs(c) > 0.1:  # Показуємо тільки значні коефіцієнти
        ao_label = mol.ao_labels()[i]
        print(f"  {ao_label:15s}: {c:8.4f}")

# LUMO (перша віртуальна)
lumo_coeff = mf.mo_coeff[0][:, mol.nelec[0]]
print("\nLUMO (σ_u* антизв'язуюча):")
for i, c in enumerate(lumo_coeff):
    if abs(c) > 0.1:
        ao_label = mol.ao_labels()[i]
        print(f"  {ao_label:15s}: {c:8.4f}")

# Матриця густини
dm = mf.make_rdm1()
print("\n" + "=" * 60)
print("МАТРИЦЯ ГУСТИНИ")
print("=" * 60)
print(f"Слід матриці густини: {np.trace(dm[0] + dm[1]):.6f}")
print(f"Очікується: {mol.nelectron:.0f} електронів")

# Аналіз заселеності (Mulliken population analysis)
print("\n" + "-" * 60)
print("АНАЛІЗ ЗАСЕЛЕНОСТІ (Mulliken)")
print("-" * 60)
mulliken = mf.mulliken_pop()
print(f"Заселеність на атомі H1: {mulliken[1][0]:.4f} e⁻")
print(f"Заселеність на атомі H2: {mulliken[1][1]:.4f} e⁻")
print(f"Сума: {sum(mulliken[1]):.4f} e⁻")

# Дипольний момент
dip = mf.dip_moment(unit="Debye")
print("\n" + "-" * 60)
print(f"Дипольний момент: {np.linalg.norm(dip):.4f} D")
print(f"Компоненти: μ_x = {dip[0]:.4f}, μ_y = {dip[1]:.4f}, μ_z = {dip[2]:.4f} D")
print("(Для гомоядерної молекули очікується ≈ 0)")

print("=" * 60)

# Візуалізація МО вздовж осі z
print("\nПобудова графіків МО...")
z = np.linspace(-5, 5, 200)
coords = np.zeros((len(z), 3))
coords[:, 2] = z

# HOMO
homo_values = tools.cubegen.orbital(
    mol, "homo_alpha.cube", mf.mo_coeff[0][:, mol.nelec[0] - 1]
)
# Спрощена візуалізація: обчислюємо значення МО вздовж осі
from pyscf.dft import numint

homo_on_grid = numint.eval_ao(mol, coords) @ homo_coeff

# LUMO
lumo_on_grid = numint.eval_ao(mol, coords) @ lumo_coeff

# Графік
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(z, homo_on_grid, "b-", linewidth=2)
ax1.axhline(0, color="k", linestyle="--", alpha=0.3)
ax1.axvline(-1, color="r", linestyle=":", alpha=0.5, label="H1")
ax1.axvline(1, color="r", linestyle=":", alpha=0.5, label="H2")
ax1.set_ylabel("HOMO (σ_g)", fontsize=12)
ax1.set_title("Молекулярні орбіталі H$_2^+$ вздовж осі z", fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(z, lumo_on_grid, "r-", linewidth=2)
ax2.axhline(0, color="k", linestyle="--", alpha=0.3)
ax2.axvline(-1, color="r", linestyle=":", alpha=0.5)
ax2.axvline(1, color="r", linestyle=":", alpha=0.5)
ax2.set_xlabel("z (bohr)", fontsize=12)
ax2.set_ylabel("LUMO (σ_u*)", fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("h2plus_mo_plots.png", dpi=300, bbox_inches="tight")
print("Графіки МО збережено: h2plus_mo_plots.png")
plt.show()
