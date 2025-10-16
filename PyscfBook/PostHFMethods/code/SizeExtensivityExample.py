from pyscf import gto, scf, ci, cc

def demonstrate_size_extensivity():
    """
    Демонстрація size-extensivity
    """

    print('\nДемонстрація size-extensivity')
    print('='*70)

    # Один атом He
    mol_1he = gto.M(atom='He 0 0 0', basis='sto-3g', verbose=0)
    mf_1he = scf.RHF(mol_1he)
    mf_1he.verbose = 0
    e_hf_1he = mf_1he.kernel()

    # CISD для 1 He
    myci_1he = ci.CISD(mf_1he)
    myci_1he.verbose = 0
    e_cisd_1he, _ = myci_1he.kernel()

    # CCSD для 1 He
    mycc_1he = cc.CCSD(mf_1he)
    mycc_1he.verbose = 0
    e_ccsd_corr_1he, _, _ = mycc_1he.kernel()
    e_ccsd_1he = e_hf_1he + e_ccsd_corr_1he

    print(f'Один атом He:')
    print(f'  HF:   {e_hf_1he:.10f} Ha')
    print(f'  CISD: {e_cisd_1he:.10f} Ha')
    print(f'  CCSD: {e_ccsd_1he:.10f} Ha')

    # Два атоми He (далеко один від одного)
    mol_2he = gto.M(
        atom='He 0 0 0; He 0 0 100',  # 100 Bohr відстань
        basis='sto-3g',
        verbose=0
    )
    mf_2he = scf.RHF(mol_2he)
    mf_2he.verbose = 0
    e_hf_2he = mf_2he.kernel()

    # CISD для 2 He
    myci_2he = ci.CISD(mf_2he)
    myci_2he.verbose = 0
    e_cisd_2he, _ = myci_2he.kernel()

    # CCSD для 2 He
    mycc_2he = cc.CCSD(mf_2he)
    mycc_2he.verbose = 0
    e_ccsd_corr_2he, _, _ = mycc_2he.kernel()
    e_ccsd_2he = e_hf_2he + e_ccsd_corr_2he

    print(f'\nДва атоми He (відокремлені):')
    print(f'  HF:   {e_hf_2he:.10f} Ha')
    print(f'  CISD: {e_cisd_2he:.10f} Ha')
    print(f'  CCSD: {e_ccsd_2he:.10f} Ha')

    # Перевірка size-extensivity
    print(f'\nПеревірка size-extensivity:')
    print(f'  2×E(1 He):')
    print(f'    HF:   {2*e_hf_1he:.10f} Ha')
    print(f'    CISD: {2*e_cisd_1he:.10f} Ha')
    print(f'    CCSD: {2*e_ccsd_1he:.10f} Ha')

    print(f'\n  Похибка E(2 He) - 2×E(1 He):')
    err_hf = (e_hf_2he - 2*e_hf_1he) * 1000
    err_cisd = (e_cisd_2he - 2*e_cisd_1he) * 1000
    err_ccsd = (e_ccsd_2he - 2*e_ccsd_1he) * 1000

    print(f'    HF:   {err_hf:.6f} mHa (має бути ≈0)')
    print(f'    CISD: {err_cisd:.6f} mHa (не size-extensive!)')
    print(f'    CCSD: {err_ccsd:.6f} mHa (має бути ≈0)')

    print('\nВисновок: CCSD є size-extensive, CISD - ні')

demonstrate_size_extensivity()

