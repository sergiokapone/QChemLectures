"""
Розрахунок іона молекули водню H2+ при фіксованій відстані
"""

from pyscf import gto, scf

# Визначення молекули
mol = gto.Mole()
mol.atom = """
    H  0.0  0.0  0.0
    H  0.0  0.0  0.74
"""
mol.basis = "sto-3g"
mol.charge = 1  # Заряд +1 (один електрон видалено)
mol.spin = 1  # 2S = N_alpha - N_beta = 1 (один непарний електрон)
mol.unit = "Angstrom"
mol.build()

# UHF розрахунок (необхідний для систем з непарними електронами)
mf = scf.UHF(mol)
energy = mf.kernel()

print("\n" + "=" * 60)
print("Іон молекули водню H2+")
print("=" * 60)
print(f"Міжядерна відстань R = 0.74 Å = {0.74 / 0.529:.3f} bohr")
print(f"Базисний набір: {mol.basis}")
print(f"Повна енергія: {energy:.6f} Ha")
print(f"Кількість електронів: α={mol.nelec[0]}, β={mol.nelec[1]}")

# Аналіз орбітальних енергій
print("\nОрбітальні енергії (Ha):")
print(f"  α-спін HOMO: {mf.mo_energy[0][0]:.4f}")
print(f"  α-спін LUMO: {mf.mo_energy[0][1]:.4f}")
print(f"  HOMO-LUMO gap: {(mf.mo_energy[0][1] - mf.mo_energy[0][0]):.4f} Ha")
print(
    f"                  = {(mf.mo_energy[0][1] - mf.mo_energy[0][0]) * 27.211:.2f} eV"
)

# Перевірка спінового стану
s2 = mf.spin_square()[0]
print(f"\n⟨S^2⟩ = {s2:.4f} (очікується {0.75:.2f} для S=1/2)")
print("=" * 60)
