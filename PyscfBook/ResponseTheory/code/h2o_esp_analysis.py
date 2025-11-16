# ============================================================
# h2o_esp_analysis.py
# Аналіз електростатичного потенціалу (ESP)
# ============================================================

from pyscf import gto, scf
import numpy as np
import matplotlib.pyplot as plt

mol = gto.M(
    atom="""
    O  0.0000  0.0000  0.1173
    H  0.0000  0.7572 -0.4692
    H  0.0000 -0.7572 -0.4692
    """,
    basis="6-31g*",
    unit="angstrom",
)

print("Електростатичний потенціал (ESP) H2O")
print("=" * 60)

# SCF розрахунок
mf = scf.RHF(mol)
mf.verbose = 0
mf.kernel()

# Створюємо сітку точок навколо молекули
# Площина XZ (де лежить молекула)
x = np.linspace(-3, 3, 100)
z = np.linspace(-3, 3, 100)
X, Z = np.meshgrid(x, z)
Y = np.zeros_like(X)

# Координати точок
coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

print("\nОбчислення ESP на сітці 100×100 точок...")

# ESP складається з ядерного та електронного внеску
# V(r) = Σ_A Z_A/|r-R_A| - ∫ ρ(r')/|r-r'| dr'

from pyscf.dft import numint

# Ядерний внесок
esp_nuc = np.zeros(len(coords))
for ia in range(mol.natm):
    Z = mol.atom_charge(ia)
    R = mol.atom_coord(ia)
    r_diff = coords - R
    r_dist = np.linalg.norm(r_diff, axis=1)
    esp_nuc += Z / r_dist

# Електронний внесок
dm = mf.make_rdm1()
esp_elec = np.zeros(len(coords))

# Обчислюємо через базисні функції
for i, coord in enumerate(coords):
    # Значення AO в точці
    ao_value = numint.eval_ao(mol, coord.reshape(1, 3))
    # ρ(r) = Σ_μν D_μν φ_μ(r) φ_ν(r)
    rho = np.einsum('pi,ij,pj->p', ao_value, dm, ao_value)[0]

    # Електронний внесок (наближено)
    esp_elec[i] = -rho / (np.linalg.norm(coord) + 1e-10)

esp_total = esp_nuc + esp_elec
esp_total = esp_total.reshape(X.shape)

# Візуалізація
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Контурна карта ESP
levels = np.linspace(-0.1, 0.1, 20)
contour = ax1.contourf(X, Z, esp_total, levels=levels, cmap='RdBu_r')
ax1.contour(X, Z, esp_total, levels=levels, colors='k', linewidths=0.5, alpha=0.3)

# Позиції атомів
for ia in range(mol.natm):
    coord = mol.atom_coord(ia)
    symbol = mol.atom_symbol(ia)
    color = 'red' if symbol == 'O' else 'lightblue'
    ax1.plot(coord[0], coord[2], 'o', color=color, markersize=12,
             markeredgecolor='k', markeredgewidth=1.5)
    ax1.text(coord[0], coord[2]-0.3, symbol, ha='center', fontsize=11, fontweight='bold')

ax1.set_xlabel('x (Å)', fontsize=12)
ax1.set_ylabel('z (Å)', fontsize=12)
ax1.set_title('Карта ESP H₂O в площині XZ', fontsize=13)
ax1.set_aspect('equal')
cbar1 = plt.colorbar(contour, ax=ax1)
cbar1.set_label('ESP (au)', fontsize=11)

# Переріз вздовж осі z
z_line = np.linspace(-3, 3, 200)
coords_line = np.column_stack([np.zeros(200), np.zeros(200), z_line])

esp_line = np.zeros(len(coords_line))
for ia in range(mol.natm):
    Z = mol.atom_charge(ia)
    R = mol.atom_coord(ia)
    r_diff = coords_line - R
    r_dist = np.linalg.norm(r_diff, axis=1)
    esp_line += Z / (r_dist + 1e-10)

ax2.plot(z_line, esp_line, 'b-', linewidth=2)
ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('z (Å)', fontsize=12)
ax2.set_ylabel('ESP (au)', fontsize=12)
ax2.set_title('ESP вздовж осі z', fontsize=13)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('h2o_esp_map.pdf', dpi=300)
print("\nКарту ESP збережено у файл h2o_esp_map.pdf")

print("\nІнтерпретація:")
print("- Негативний ESP (червоний) біля O: нуклеофільна область")
print("- Позитивний ESP (синій) біля H: електрофільна область")
print("- Використовується для передбачення реакційної здатності")

