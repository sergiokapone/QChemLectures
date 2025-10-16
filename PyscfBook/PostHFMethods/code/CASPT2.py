from pyscf import gto, scf, mcscf, mrpt


def caspt2_calculation(symbol="Be", spin=0, nelecas=4, ncas=8, basis="cc-pvdz"):
    """
    CASSCF + CASPT2 розрахунок
    """

    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

    print(f"\nCASPT2 розрахунок {symbol}")
    print(f"Активний простір: ({nelecas},{ncas})")
    print("=" * 70)

    # HF
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)

    mf.verbose = 0
    e_hf = mf.kernel()

    print(f"HF енергія: {e_hf:.10f} Ha")

    # CASSCF
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.verbose = 0
    mc.conv_tol = 1e-9
    e_casscf = mc.kernel()[0]

    print(f"CASSCF енергія: {e_casscf:.10f} Ha")
    print(f"Статична кореляція: {(e_casscf - e_hf) * 1000:.6f} mHa")

    # CASPT2
    print("\nCASPT2 розрахунок...")
    try:
        # В PySCF NEVPT2 більш стабільний ніж CASPT2
        from pyscf.mrpt import NEVPT

        nevpt2 = NEVPT(mc)
        e_corr_pt2 = nevpt2.kernel()
        e_caspt2 = e_casscf + e_corr_pt2

        print(f"NEVPT2 кореляція: {e_corr_pt2 * 1000:.6f} mHa")
        print(f"CASSCF+NEVPT2 енергія: {e_caspt2:.10f} Ha")

        # Розбиття кореляції
        total_corr = e_caspt2 - e_hf
        static_corr = e_casscf - e_hf
        dynamic_corr = e_corr_pt2

        print(f"\nРозбиття кореляційної енергії:")
        print(f"  Повна: {total_corr * 1000:.6f} mHa")
        print(
            f"  Статична (CASSCF): {static_corr * 1000:.6f} mHa "
            f"({abs(static_corr / total_corr) * 100:.1f}%)"
        )
        print(
            f"  Динамічна (NEVPT2): {dynamic_corr * 1000:.6f} mHa "
            f"({abs(dynamic_corr / total_corr) * 100:.1f}%)"
        )

        return e_hf, e_casscf, e_caspt2

    except Exception as e:
        print(f"CASPT2/NEVPT2 недоступний: {str(e)}")
        return e_hf, e_casscf, None


# Приклади
caspt2_calculation("Be", spin=0, nelecas=4, ncas=8)
caspt2_calculation("C", spin=2, nelecas=4, ncas=4)
