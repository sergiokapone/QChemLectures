from pyscf import gto, dft

# Тестування на атомі Кисню
mol = gto.M(atom="O 0 0 0", basis="def2-tzvp", spin=2)

gga_functionals = ["pbe", "blyp", "bp86", "pw91", "pberev"]

print("Порівняння GGA функціоналів для атома O (³P)")
print("=" * 60)
print(f"{'Функціонал':12s} {'Енергія, Ha':15s} {'Відносно PBE, mHa':20s}")
print("-" * 60)

energies = {}

for xc in gga_functionals:
    mf = dft.UKS(mol)
    mf.xc = xc
    mf.verbose = 0
    mf.conv_tol = 1e-10

    try:
        energy = mf.kernel()
        energies[xc] = energy

        if xc == "pbe":
            e_ref = energy
            rel = 0.0
        else:
            rel = (energy - e_ref) * 1000

        print(f"{xc:12s} {energy:15.8f} {rel:20.4f}")
    except:
        print(f"{xc:12s} --- помилка розрахунку")

print("=" * 60)
