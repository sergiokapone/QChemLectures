from pyscf import gto, scf


def transition_metal_calculation(symbol, spin, basis="def2-svp"):
    """
    Розрахунок атома перехідного металу
    Часто потребує спеціальних налаштувань
    """

    print(f"\nРозрахунок {symbol} (2S={spin})")
    print("=" * 60)

    mol = gto.M(
        atom=f"{symbol} 0 0 0",
        basis=basis,
        spin=spin,
        symmetry=False,  # Іноді краще без симетрії
        verbose=0,
    )

    mf = scf.UHF(mol)

    # Налаштування для важких випадків
    mf.conv_tol = 1e-8
    mf.max_cycle = 200
    mf.level_shift = 0.5  # Level shift допомагає конвергенції
    mf.diis_space = 12
    mf.init_guess = "atom"

    print("Спроба 1: UHF з level shift...")
    mf.verbose = 4
    energy = mf.kernel()

    if not mf.converged:
        print("\nНе конвергувало! Спроба 2: Newton-Raphson...")
        mf = mf.newton()
        mf.max_cycle = 50
        energy = mf.kernel()

    if mf.converged:
        s2 = mf.spin_square()
        expected_s2 = spin * (spin + 2) / 4

        print(f"\nРезультати:")
        print(f"  Енергія: {energy:.8f} Ha")
        print(f"  <S²>: {s2[0]:.4f} (очікується {expected_s2:.4f})")
        print(f"  Забруднення: {s2[0] - expected_s2:.4f}")
    else:
        print("\nНЕ ВДАЛОСЯ ДОСЯГТИ КОНВЕРГЕНЦІЇ!")

    return mf, energy if mf.converged else None


# Приклади перехідних металів
# Sc: [Ar] 3d¹ 4s², ²D
mf_sc, e_sc = transition_metal_calculation("Sc", spin=1)

# Ti: [Ar] 3d² 4s², ³F
mf_ti, e_ti = transition_metal_calculation("Ti", spin=2)

# Cr: [Ar] 3d⁵ 4s¹, ⁷S
mf_cr, e_cr = transition_metal_calculation("Cr", spin=6)

# Mn: [Ar] 3d⁵ 4s², ⁶S
mf_mn, e_mn = transition_metal_calculation("Mn", spin=5)

# Fe: [Ar] 3d⁶ 4s², ⁵D
mf_fe, e_fe = transition_metal_calculation("Fe", spin=4)
