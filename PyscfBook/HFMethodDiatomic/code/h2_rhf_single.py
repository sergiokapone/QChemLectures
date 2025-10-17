"""
RHF розрахунок молекули водню H2 при рівноважній відстані
"""

import numpy as np
from pyscf import gto, scf

# Молекула H2 при рівноважній відстані
mol = gto.Mole()
mol.atom = """
    H  0.0  0.0  0.0
    H  0.0  0.0  0.74
"""
mol.basis = "cc-pvdz"
mol.charge = 0  # Нейтральна молекула
mol.spin = 0  # Синглет (замкнена оболонка)
mol.unit = "Angstrom"
mol.build()

print("\n" + "=" * 60)
print("МОЛЕКУЛА ВОДНЮ H2 (RHF)")
print("=" * 60)

# RHF розрахунок
mf = scf.RHF(mol)
E_rhf = mf.kernel()

print(f"\nМіжядерна відстань R = 0.74 Å = {0.74 / 0.529177:.3f} bohr")
print(f"Базисний набір: {mol.basis}")
print(f"Кількість електронів: {mol.nelectron}")
print(f"Кількість базисних функцій: {mol.nao}")

print("\n" + "-" * 60)
print("ЕНЕРГЕТИЧНІ КОМПОНЕНТИ:")
print("-" * 60)

# Компоненти енергії
dm = mf.make_rdm1()
h1e = mf.get_hcore()
vhf = mf.get_veff()

E_kin = np.einsum("ij,ji->", dm, mol.intor("int1e_kin"))
E_nuc = np.einsum("ij,ji->", dm, mol.intor("int1e_nuc"))
E_ee = 0.5 * np.einsum("ij,ji->", dm, vhf)
E_nn = mol.energy_nuc()

print(f"Кінетична енергія T:           {E_kin:12.6f} Ha")
print(f"Електрон-ядро взаємодія V_ne:  {E_nuc:12.6f} Ha")
print(f"Електрон-електрон V_ee:        {E_ee:12.6f} Ha")
print(f"Ядро-ядро відштовхування V_nn: {E_nn:12.6f} Ha")
print(f"{'─' * 60}")
print(f"Повна електронна енергія:      {E_kin + E_nuc + E_ee:12.6f} Ha")
print(f"Повна енергія (з V_nn):        {E_rhf:12.6f} Ha")

# Віріальна теорема
virial_ratio = -(E_nuc + E_ee + E_nn) / E_kin
print("\n" + "-" * 60)
print("ВІРІАЛЬНА ТЕОРЕМА:")
print("-" * 60)
print(f"⟨V⟩/⟨T⟩ = {virial_ratio:.6f}")
print(f"Очікується: -2.000000 для точного розв'язку")
print(f"Відхилення: {abs(virial_ratio + 2.0) * 100:.4f}%")

# Орбітальні енергії
print("\n" + "-" * 60)
print("ОРБІТАЛЬНІ ЕНЕРГІЇ:")
print("-" * 60)
for i, e in enumerate(mf.mo_energy):
    occ_str = "зайнята" if i < mol.nelec[0] else "віртуальна"
    print(f"МО {i + 1}: ε = {e:10.6f} Ha = {e * 27.2114:10.4f} eV  ({occ_str})")

# HOMO-LUMO gap
homo_energy = mf.mo_energy[mol.nelec[0] - 1]
lumo_energy = mf.mo_energy[mol.nelec[0]]
gap = lumo_energy - homo_energy

print(f"\nHOMO: ε = {homo_energy:.6f} Ha")
print(f"LUMO: ε = {lumo_energy:.6f} Ha")
print(f"HOMO-LUMO gap: {gap:.6f} Ha = {gap * 27.2114:.4f} eV")

# Порівняння з експериментом
print("\n" + "=" * 60)
print("ПОРІВНЯННЯ З ЕКСПЕРИМЕНТОМ:")
print("=" * 60)
E_exp = -1.174  # Ha (точна енергія Full CI)
D_e_exp = 4.75  # eV

# Енергія дисоціації
E_2H = 2 * (-0.5)  # 2 × E(H)
D_e_rhf = E_2H - E_rhf

print(f"Енергія H2 (RHF):        {E_rhf:.6f} Ha")
print(f"Енергія H2 (точна):      {E_exp:.6f} Ha")
print(
    f"Кореляційна енергія:     {E_exp - E_rhf:.6f} Ha = {(E_exp - E_rhf) * 27.2114:.3f} eV"
)
print(f"% від повної енергії:    {abs((E_exp - E_rhf) / E_exp) * 100:.2f}%")

print(f"\nD_e (RHF):               {D_e_rhf:.6f} Ha = {D_e_rhf * 27.2114:.3f} eV")
print(f"D_e (експеримент):       ---          = {D_e_exp:.2f} eV")
print(f"Похибка:                 {abs(D_e_rhf * 27.2114 - D_e_exp):.3f} eV")

# Дипольний момент
dip = mf.dip_moment(unit="Debye")
print(f"\nДипольний момент:        {np.linalg.norm(dip):.6f} D")
print("(Для гомоядерної молекули = 0)")

print("=" * 60)
