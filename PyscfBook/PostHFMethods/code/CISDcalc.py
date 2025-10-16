from pyscf import gto, scf, ci

def cisd_calculation(symbol, spin, basis='cc-pvtz'):
    """
    CISD розрахунок
    """

    mol = gto.M(
        atom=f'{symbol} 0 0 0',
        basis=basis,
        spin=spin,
        verbose=0
    )

    print(f'\nCISD розрахунок {symbol} (базис: {basis})')
    print('='*70)

    # HF
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)

    mf.verbose = 0
    mf.conv_tol = 1e-11
    e_hf = mf.kernel()

    print(f'HF енергія: {e_hf:.10f} Ha')

    # CISD
    print('\nCISD розрахунок...')
    if spin == 0:
        myci = ci.CISD(mf)
    else:
        myci = ci.UCISD(mf)

    myci.verbose = 4
    e_cisd, civec = myci.kernel()

    print(f'\nCISD енергія: {e_cisd:.10f} Ha')
    print(f'CISD кореляція: {(e_cisd - e_hf)*1000:.6f} mHa')

    return e_hf, e_cisd

cisd_calculation('C', spin=2)
cisd_calculation('Ne', spin=0)
