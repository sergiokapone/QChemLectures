from pyscf import gto, scf

# Важкий випадок: атом з близькими за енергією орбіталями
mol = gto.M(atom="V 0 0 0", basis="cc-pvdz", spin=3, verbose=0)

print("Розрахунок V з дробовими заповненнями")
print("=" * 60)

# Стандартний UHF може не конвергувати
mf = scf.UHF(mol)
mf.verbose = 0
mf.max_cycle = 100

try:
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError("Не конвергувало")
except:
    print("Стандартний UHF не конвергував")

# Використання дробових заповнень (smearing)
print("\nСпроба з дробовими заповненнями...")

mf = scf.UHF(mol)
mf = scf.addons.frac_occ(mf)
mf.verbose = 4
energy = mf.kernel()

if mf.converged:
    print(f"\nУспішно! Енергія: {energy:.8f} Ha")
else:
    print("\nВсе одно не конвергувало")
