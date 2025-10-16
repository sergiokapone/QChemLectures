from pyscf import gto, scf, ci, cc, fci
import matplotlib.pyplot as plt

def compare_ci_cc(symbol='Be', spin=0, basis='cc-pvdz'):
    """
    Порівняння CI та CC методів
    """
    
    mol = gto.M(
        atom=f'{symbol} 0 0 0',
        basis=basis,
        spin=spin,
        verbose=0
    )

    # Перевірка розміру для FCI
    if mol.nelectron > 10:
        print('Атом занадто великий для FCI')
        include_fci = False
    else:
        include_fci = True

    print(f'\nПорівняння методів для {symbol} (базис: {basis})')
    print('='*70)

    results = {}

    # HF
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)

    mf.verbose = 0
    mf.conv_tol = 1e-11
    e_hf = mf.kernel()
    results['HF'] = e_hf

    print(f'HF: {e_hf:.10f} Ha')

    # CISD
    print('CISD...')
    if spin == 0:
        myci = ci.CISD(mf)
    else:
        myci = ci.UCISD(mf)

    myci.verbose = 0
    e_cisd, _ = myci.kernel()
    results['CISD'] = e_cisd

    print(f'CISD: {e_cisd:.10f} Ha')

    # CCSD
    print('CCSD...')
    if spin == 0:
        mycc = cc.CCSD(mf)
    else:
        mycc = cc.UCCSD(mf)

    mycc.verbose = 0
    e_ccsd_corr, _, _ = mycc.kernel()
    e_ccsd = e_hf + e_ccsd_corr
    results['CCSD'] = e_ccsd

    print(f'CCSD: {e_ccsd:.10f} Ha')

    # CCSD(T)
    print('CCSD(T)...')
    e_t = mycc.ccsd_t()
    e_ccsdt = e_ccsd + e_t
    results['CCSD(T)'] = e_ccsdt

    print(f'CCSD(T): {e_ccsdt:.10f} Ha')

    # FCI
    if include_fci:
        print('FCI...')
        myfci = fci.FCI(mf)
        myfci.verbose = 0
        e_fci, _ = myfci.kernel()
        results['FCI'] = e_fci

        print(f'FCI: {e_fci:.10f} Ha')

    # Порівняння
    print('\n' + '='*70)
    print(f'{"Метод":12s} {"Енергія, Ha":18s} {"Кореляція, mHa":18s}')
    print('-'*70)

    for method in results.keys():
        e = results[method]
        corr = (e - e_hf) * 1000
        print(f'{method:12s} {e:18.10f} {corr:18.6f}')

    print('='*70)

    # Аналіз
    if include_fci:
        e_corr_fci = results['FCI'] - e_hf

        print('\nВідсоток відновленої кореляції (відносно FCI):')
        for method in ['CISD', 'CCSD', 'CCSD(T)']:
            e_corr = results[method] - e_hf
            percent = abs(e_corr / e_corr_fci) * 100
            print(f'{method:12s}: {percent:6.2f}%')

        # Графік
        fig, ax = plt.subplots(figsize=(10, 6))

        methods_list = list(results.keys())
        energies = [results[m] for m in methods_list]
        corr_energies = [(results[m] - e_hf) * 1000
                        for m in methods_list]

        x = np.arange(len(methods_list))
        colors = ['blue', 'orange', 'green', 'red', 'purple']

        bars = ax.bar(x, corr_energies, color=colors[:len(methods_list)],
                     alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(methods_list)
        ax.set_ylabel('Кореляційна енергія (mHa)', fontsize=12)
        ax.set_title(f'Порівняння методів для {symbol}', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        # Додавання значень
        for bar, e in zip(bars, corr_energies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{e:.4f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{symbol}_ci_cc_comparison.pdf')
        plt.show()

    return results

# Приклади
results_be = compare_ci_cc('Be', spin=0, basis='cc-pvdz')
results_c = compare_ci_cc('C', spin=2, basis='6-31g')
