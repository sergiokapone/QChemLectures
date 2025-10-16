from pyscf import gto, scf, mcscf


def sa_casscf_carbon():
    """
    State-averaged CASSCF для різних станів Карбону
    """

    mol = gto.M(
        atom="C 0 0 0",
        basis="cc-pvtz",
        spin=2,  # Для триплету
        symmetry=True,
        verbose=0,
    )

    print("\nState-Averaged CASSCF для C")
    print("Стани: ³P (основний) та ¹D (збуджений)")
    print("=" * 70)

    # HF
    mf = scf.UHF(mol)
    mf.verbose = 0
    e_hf = mf.kernel()

    print(f"UHF енергія: {e_hf:.10f} Ha")

    # SA-CASSCF(4,4): усереднення по декількох станах
    mc = mcscf.CASSCF(mf, 4, 4)
    mc.verbose = 4

    # State-averaging для 2 станів з рівними вагами
    mc.state_average_([0.5, 0.5])

    # Або для різних симетрій:
    # mc = mc.state_average_mix_([
    #     mcscf.state_average_(mc, [0.5, 0.5]),  # стани симетрії 1
    # ])

    mc.conv_tol = 1e-9
    e_sa = mc.kernel()[0]

    print(f"\nSA-CASSCF енергія (усереднена): {e_sa:.10f} Ha")

    # Енергії окремих станів
    print("\nЕнергії окремих станів:")
    for i, e_state in enumerate(mc.e_states):
        print(f"  Стан {i + 1}: {e_state:.10f} Ha ({e_state * 27.211386:.4f} eV)")

    # Різниця енергій
    if len(mc.e_states) > 1:
        delta_e = (mc.e_states[1] - mc.e_states[0]) * 27.211386
        print(f"\nРізниця енергій збудження: {delta_e:.4f} eV")


sa_casscf_carbon()
