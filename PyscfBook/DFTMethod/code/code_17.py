from pyscf import gto, dft


def transition_metal_dft(symbol, spin, functional="tpss", basis="def2-tzvp"):
    """
    DFT розрахунок атома перехідного металу
    """

    print(f"\nРозрахунок {symbol} (2S={spin}, {functional.upper()})")
    print("=" * 60)

    mol = gto.M(
        atom=f"{symbol} 0 0 0",
        basis=basis,
        spin=spin,
        symmetry=False,  # Часто краще без симетрії
        verbose=0,
    )

    mf = dft.UKS(mol)
    mf.xc = functional

    # Налаштування для важких випадків
    mf.conv_tol = 1e-8
    mf.max_cycle = 200
    mf.diis_space = 12

    # Для перехідних металів часто потрібен level shift
    if symbol in ["Cr", "Mn", "Fe", "Co", "Ni"]:
        mf.level_shift = 0.3

    mf.verbose = 4
    energy = mf.kernel()

    if mf.converged:
        s2 = mf.spin_square()
        expected_s2 = spin * (spin + 2) / 4

        print(f"\nРезультати:")
        print(f"  Енергія: {energy:.8f} Ha")
        print(f"  <S²>: {s2[0]:.4f} (очікується {expected_s2:.4f})")
        print(f"  Забруднення спіном: {s2[0] - expected_s2:.4f}")

        # Заселеності d-орбіталей
        from pyscf import lo

        pop = mf.mulliken_pop()

        return energy, s2[0]
    else:
        print("\nНе конвергувало!")
        return None, None


# Приклади 3d металів
# Sc: [Ar] 3d¹ 4s², ²D
e_sc, s2_sc = transition_metal_dft("Sc", spin=1, functional="pbe")

# Ti: [Ar] 3d² 4s², ³F
e_ti, s2_ti = transition_metal_dft("Ti", spin=2, functional="pbe")

# V: [Ar] 3d³ 4s², ⁴F
e_v, s2_v = transition_metal_dft("V", spin=3, functional="pbe")

# Cr: [Ar] 3d⁵ 4s¹, ⁷S (виняток!)
e_cr, s2_cr = transition_metal_dft("Cr", spin=6, functional="pbe")

# Mn: [Ar] 3d⁵ 4s², ⁶S
e_mn, s2_mn = transition_metal_dft("Mn", spin=5, functional="pbe")

# Fe: [Ar] 3d⁶ 4s², ⁵D
e_fe, s2_fe = transition_metal_dft("Fe", spin=4, functional="pbe")
