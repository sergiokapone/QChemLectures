from pyscf import gto, dft


def heavy_atom_calculation(
    symbol, spin, functional="pbe", basis="def2-svp", relativistic=False
):
    """
    Розрахунок важкого атома з/без релятивістських ефектів
    """

    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

    mf = dft.UKS(mol)
    mf.xc = functional

    if relativistic:
        # X2C (exact 2-component) релятивістський гамільтоніан
        mf = mf.x2c()
        print(f"\nРозрахунок {symbol} з X2C релятивістськими корекціями")
    else:
        print(f"\nРозрахунок {symbol} (нерелятивістський)")

    mf.conv_tol = 1e-9
    mf.max_cycle = 150
    energy = mf.kernel()

    print(f"Енергія: {energy:.8f} Ha")

    return energy


# Порівняння релятивістських ефектів
print("Порівняння релятивістських ефектів")
print("=" * 70)

# 5d метал: Золото
print("\nЗолото (Au):")
e_au_nr = heavy_atom_calculation("Au", spin=1, relativistic=False)
e_au_r = heavy_atom_calculation("Au", spin=1, relativistic=True)
rel_effect = (e_au_r - e_au_nr) * 627.509  # kcal/mol
print(f"Релятивістський ефект: {rel_effect:.2f} kcal/mol")

# 5d метал: Платина
print("\nПлатина (Pt):")
e_pt_nr = heavy_atom_calculation("Pt", spin=2, relativistic=False)
e_pt_r = heavy_atom_calculation("Pt", spin=2, relativistic=True)
rel_effect = (e_pt_r - e_pt_nr) * 627.509
print(f"Релятивістський ефект: {rel_effect:.2f} kcal/mol")
