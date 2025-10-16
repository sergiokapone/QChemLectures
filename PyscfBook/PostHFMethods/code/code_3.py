from pyscf import gto, scf, mcscf


def multiconfigurational_demo():
    """
    Демонстрація багатоконфігураційного характеру
    """

    # Атом Be: [He] 2s²
    # Близькі за енергією 2s² та 2p²

    mol = gto.M(atom="Be 0 0 0", basis="cc-pvtz", spin=0, verbose=0)

    print("Багатоконфігураційний характер Be")
    print("=" * 60)

    # RHF
    mf = scf.RHF(mol)
    mf.verbose = 0
    e_hf = mf.kernel()

    print(f"RHF енергія: {e_hf:.8f} Ha")

    # CASSCF(4,8): 4 електрони у 8 орбіталях (2s, 2p, 3s, 3p)
    mc = mcscf.CASSCF(mf, 8, 4)
    mc.verbose = 0
    e_casscf = mc.kernel()[0]

    print(f"CASSCF(4,8) енергія: {e_casscf:.8f} Ha")
    print(f"Статична кореляція: {(e_casscf - e_hf) * 1000:.4f} mHa")

    # Аналіз конфігурацій
    print(f"\nОсновні конфігурації (вага > 1%):")

    # Отримання CI коефіцієнтів
    ci_coeff = mc.ci

    # Для детального аналізу потрібен додатковий код
    print("(детальний аналіз потребує додаткової обробки)")

    print("=" * 60)


multiconfigurational_demo()
