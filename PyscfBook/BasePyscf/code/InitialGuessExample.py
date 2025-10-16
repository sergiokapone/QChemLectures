from pyscf import gto, scf

mol = gto.M(atom='Cr 0 0 0', basis='cc-pvdz', spin=6)
mf = scf.UHF(mol)

# Метод 1: Hcore (за замовчуванням)
mf.init_guess = 'hcore'  # Використання Гамільтоніана ядра

# Метод 2: Мінімальний базис (розширене наближення)
mf.init_guess = 'minao'

# Метод 3: Атомні густини (оптимально для атомів та іонів)
mf.init_guess = 'atom'

# Метод 4: 1e guess (використання лише одноелектронних інтегралів)
mf.init_guess = '1e'

# Метод 5: З попереднього файлу (для restart)
# mf.init_guess = 'chkfile'
# mf.chkfile = 'previous_calc.chk'

energy = mf.kernel()
