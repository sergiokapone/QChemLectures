from pyscf import gto, scf

mol = gto.M(atom='Fe 0 0 0', basis='def2-svp', spin=4)
mf = scf.UHF(mol)

# Критерій конвергенції (допустима різниця енергії між ітераціями)
mf.conv_tol = 1e-10  # За замовчуванням 1e-9

# Максимальна кількість ітерацій SCF
mf.max_cycle = 100   # За замовчуванням 50

# Використання DIIS (Pulay mixing) для прискорення збіжності
mf.diis = True       # Активовано за замовчуванням
mf.diis_space = 8    # Розмір DIIS-простору (типово 6–8)

# Level shift — підйом незайнятих орбіталей для стабілізації
mf.level_shift = 0.5  # У Hartree, типово 0.3–1.0 для важких випадків

energy = mf.kernel()
