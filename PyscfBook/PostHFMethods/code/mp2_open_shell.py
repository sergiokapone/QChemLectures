from pyscf import gto, scf, mp


def ump2_calculation(symbol, spin, basis="cc-pvtz"):
    """
    UMP2 розрахунок для відкритої оболонки
    """

    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

    print(f"\nUMP2 розрахунок {symbol} (2S={spin})")
    print("=" * 70)

    # UHF
    mf = scf.UHF(mol)
    mf.verbose = 0
    mf.conv_tol = 1e-10
    e_hf = mf.kernel()

    print(f"UHF енергія: {e_hf:.10f} Ha")

    # Забруднення спіном
    s2_hf = mf.spin_square()[0]
    expected_s2 = spin * (spin + 2) / 4
    print(f"<S²> (UHF): {s2_hf:.6f} (очікується {expected_s2:.6f})")

    # UMP2
    mymp2 = mp.UMP2(mf)
    mymp2.verbose = 0
    e_mp2_corr, t2 = mymp2.kernel()

    e_total = e_hf + e_mp2_corr

    print(f"\nUMP2 кореляція: {e_mp2_corr:.10f} Ha")
    print(f"UMP2 повна енергія: {e_total:.10f} Ha")

    # MP2 не виправляє забруднення спіном
    # (для цього потрібні інші методи)

    return e_hf, e_total, e_mp2_corr


# Приклади
e_hf_li, e_mp2_li, _ = ump2_calculation("Li", spin=1)
e_hf_c, e_mp2_c, _ = ump2_calculation("C", spin=2)
e_hf_n, e_mp2_n, _ = ump2_calculation("N", spin=3)
e_hf_o, e_mp2_o, _ = ump2_calculation("O", spin=2)
