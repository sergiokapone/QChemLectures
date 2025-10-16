from pyscf import gto

# Створення порожнього об'єкта
mol = gto.Mole()

# Налаштування параметрів
mol.atom = 'C 0 0 0'
mol.basis = 'cc-pvdz'
mol.charge = 0
mol.spin = 2  # Два неспарені електрони

# Завершення побудови (важливо!)
mol.build()
