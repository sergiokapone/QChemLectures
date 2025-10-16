from pyscf import gto, scf, lib
import numpy as np

def analyze_atom(symbol, charge=0, spin=None, basis='cc-pvdz'):
    """
    Комплексний аналіз атома

    Parameters:
    -----------
    symbol : str
        Символ атома
    charge : int
        Заряд (0 для нейтрального)
    spin : int
        2S (якщо None, визначається автоматично)
    basis : str
        Базисний набір
    """

    print(f'\n{"="*60}')
    print(f'Аналіз атома {symbol} (charge={charge}, basis={basis})')
    print(f'{"="*60}\n')

    # Створення молекулярного об'єкта
    mol = gto.M(
        atom=f'{symbol} 0 0 0',
        basis=basis,
        charge=charge,
        spin=spin if spin is not None else 0,
        symmetry=True,
        verbose=4
    )

    # Інформація про систему
    print(f'Кількість електронів: {mol.nelectron}')
    print(f'α-електронів: {mol.nelec[0]}')
    print(f'β-електронів: {mol.nelec[1]}')
    print(f'Спін (2S): {mol.spin}')
    print(f'Мультиплетність: {mol.spin + 1}')
    print(f'Базисних функцій: {mol.nao_nr()}')
    print(f'Точкова група: {mol.groupname}\n')

    # Вибір методу SCF
    if mol.spin == 0:
        mf = scf.RHF(mol)
        method_name = 'RHF'
    else:
        mf = scf.UHF(mol)
        method_name = 'UHF'

    # Налаштування параметрів
    mf.conv_tol = 1e-10
    mf.max_cycle = 100
    mf.init_guess = 'atom'

    # Розрахунок
    print(f'Запуск {method_name} розрахунку...\n')
    energy = mf.kernel()

    if not mf.converged:
        print('Не конвергувало! Спроба Newton-Raphson...')
        mf = mf.newton()
        energy = mf.kernel()

    # Результати
    print(f'\n{"="*60}')
    print(f'РЕЗУЛЬТАТИ')
    print(f'{"="*60}')
    print(f'Енергія: {energy:.10f} Ha')
    print(f'Енергія: {energy * 27.211386:.6f} eV')
    print(f'Конвергувало: {mf.converged}')

    # Аналіз орбіталей
    print(f'\nОрбітальні енергії ({method_name}):')

    if mol.spin == 0:
        # RHF
        print('\nЗаповнені орбіталі:')
        n_occ = mol.nelec[0]
        for i in range(n_occ):
            print(f'  MO {i+1:2d}: {mf.mo_energy[i]:10.6f} Ha '
                  f'({mf.mo_energy[i]*27.211386:8.4f} eV)')

        print('\nВіртуальні орбіталі (перші 5):')
        for i in range(n_occ, min(n_occ+5, len(mf.mo_energy))):
            print(f'  MO {i+1:2d}: {mf.mo_energy[i]:10.6f} Ha '
                  f'({mf.mo_energy[i]*27.211386:8.4f} eV)')
    else:
        # UHF
        print('\nАльфа-орбіталі (заповнені):')
        n_alpha = mol.nelec[0]
        for i in range(n_alpha):
            print(f'  α-MO {i+1:2d}: {mf.mo_energy[0][i]:10.6f} Ha '
                  f'({mf.mo_energy[0][i]*27.211386:8.4f} eV)')

        print('\nБета-орбіталі (заповнені):')
        n_beta = mol.nelec[1]
        for i in range(n_beta):
            print(f'  β-MO {i+1:2d}: {mf.mo_energy[1][i]:10.6f} Ha '
                  f'({mf.mo_energy[1][i]*27.211386:8.4f} eV)')

    # Заселеності Малікена
    print(f'\n{"="*60}')
    print('АНАЛІЗ ЗАСЕЛЕНОСТЕЙ (Mulliken)')
    print(f'{"="*60}')

    pop = mf.mulliken_pop()

    # Дипольний момент (для нейтральних атомів = 0)
    dip = mf.dip_moment(unit='Debye')
    print(f'\nДипольний момент: ({dip[0]:.4f}, {dip[1]:.4f}, '
          f'{dip[2]:.4f}) Debye')
    print(f'|μ| = {np.linalg.norm(dip):.6f} Debye')

    # Спінова густина (для відкритих систем)
    if mol.spin > 0:
        print(f'\nСпінова густина:')
        s = mf.spin_square()
        print(f'  <S²> = {s[0]:.4f}')
        print(f'  <S²> (очікуване) = {mol.spin*(mol.spin+2)/4:.4f}')
        print(f'  Забруднення спіном: {s[0] - mol.spin*(mol.spin+2)/4:.4f}')

    return mf, energy


# Приклади використання
if __name__ == '__main__':
    # Атом Li (основний стан)
    mf_li, e_li = analyze_atom('Li', spin=1, basis='cc-pvdz')

    # Атом C (триплет)
    mf_c, e_c = analyze_atom('C', spin=2, basis='6-31g*')

    # Катіон O+
    mf_o_plus, e_o_plus = analyze_atom('O', charge=1, spin=3,
                                        basis='aug-cc-pvdz')
