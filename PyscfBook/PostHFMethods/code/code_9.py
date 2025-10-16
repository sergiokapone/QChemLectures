from pyscf import gto, scf, cc


def uccsd_calculation(symbol, spin, basis="cc-pvtz"):
    """
    UCCSD розрахунок для відкритої оболонки
    """

    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

    print(f"\nUCCSD розрахунок {symbol} (2S={spin}, базис: {basis})")
    print("=" * 70)

    # UHF
    mf = scf.UHF(mol)
    mf.verbose = 0
    mf.conv_tol = 1e-11
    e_hf = mf.kernel()

    print(f"UHF енергія: {e_hf:.10f} Ha")

    # Забруднення спіном
    s2_hf = mf.spin_square()[0]
    expected_s2 = spin * (spin + 2) / 4
    spin_cont = s2_hf - expected_s2
    print(f"<S²> (UHF): {s2_hf:.6f} (очікується {expected_s2:.6f})")
    print(f"Забруднення спіном: {spin_cont:.6f}")

    # UCCSD
    print("\nUCCSD розрахунок...")
    mycc = cc.UCCSD(mf)
    mycc.verbose = 4
    mycc.conv_tol = 1e-9

    e_ccsd_corr, t1, t2 = mycc.kernel()
    e_ccsd = e_hf + e_ccsd_corr

    print(f"\nUCCSD кореляція: {e_ccsd_corr:.10f} Ha")
    print(f"UCCSD повна енергія: {e_ccsd:.10f} Ha")

    # CCSD виправляє забруднення спіном
    # (але <S²> все ще не є точним оператором)

    # Аналіз T1 діагностики
    t1_alpha, t1_beta = t1
    t1_diag = np.linalg.norm(t1_alpha) / np.sqrt(mol.nelec[0])

    print(f"\nT1 діагностика: {t1_diag:.6f}")
    if t1_diag < 0.02:
        print("  T1 < 0.02: одноконфігураційний характер")
    elif t1_diag < 0.05:
        print("  0.02 < T1 < 0.05: слабка багатоконфігураційність")
    else:
        print("  T1 > 0.05: сильна багатоконфігураційність!")
        print("  Можливо потрібен CASSCF/MRCI")

    # UCCSD(T)
    if mol.nelectron <= 10:  # (T) дуже повільно
        print("\n(T) корекція...")
        e_t = mycc.ccsd_t()
        e_ccsdt = e_ccsd + e_t

        print(f"(T) корекція: {e_t:.10f} Ha")
        print(f"UCCSD(T) повна енергія: {e_ccsdt:.10f} Ha")

        return e_hf, e_ccsd, e_ccsdt
    else:
        return e_hf, e_ccsd, None


# Приклади
uccsd_calculation("Li", spin=1, basis="cc-pvdz")
uccsd_calculation("C", spin=2, basis="cc-pvdz")
uccsd_calculation("N", spin=3, basis="cc-pvdz")
