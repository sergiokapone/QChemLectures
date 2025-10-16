from pyscf import gto, scf
# Етап 1: Створення молекулярного об'єкта
mol = gto.M(
atom='H 0 0 0',      # Координати атома
basis='sto-3g',       # Базисний набір
spin=1                # 2S (1 неспарений електрон)
)
# Етап 2: Вибір методу (UHF для відкритої оболонки)
mf = scf.UHF(mol)
# Етап 3: Виконання розрахунку
energy = mf.kernel()
# Виведення результатів
print(f'Енергія атома H: {energy:.8f} Hartree')
print(f'Енергія атома H: {energy * 27.211386:.6f} eV')