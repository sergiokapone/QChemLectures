from pyscf import gto, scf, fci, ci
import numpy as np

def fci_calculation_detailed(symbol, spin, basis='cc-pvdz'):
    """
    Детальний Full CI розрахунок
    """

    mol = gto.M(
        atom=f'{symbol} 0 0 0',
        basis=basis,
        spin=spin,
        verbose=0
    )

    # Перевірка розміру
    n_det_approx = np.math.factorial(mol.nao_nr()) / (
        np.math.factorial(mol.nelec[0]) *
        np.math.factorial(mol.nao_nr() - mol.nelec[0])
    )

    if n_det_approx > 1e8:
        print(f'Система занадто велика для FCI!')
        print(f'Приблизна кількість детермінантів: {n_det_approx:.2e}')
        return

    print(f'\nFull CI розрахунок {symbol} (базис: {basis})')
    print('='*70)

    print(f'Розмір задачі:')
    print(f'  Електронів: {mol.nelectron}')
    print(f'  Базисних функцій: {mol.nao_nr()}')
    print(f'  α-електронів: {mol.nelec[0]}')
    print(f'  β-електронів: {mol.nelec[1]}')
    print(f'  Приблизна кількість детермінантів: {n_det_approx:.2e}')

    # HF розрахунок
    print('\nHF розрахунок...')
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)

    mf.verbose = 0
    mf.conv_tol = 1e-12
    e_hf = mf.kernel()

    print(f'HF енергія: {e_hf:.12f} Ha')

    # FCI розрахунок
    print('\nFull CI розрахунок...')
    myfci = fci.FCI(mf)
    myfci.verbose = 4

    e_fci, ci_vec = myfci.kernel()

    print(f'\nFCI енергія: {e_fci:.12f} Ha')
    print(f'Кореляційна енергія: {(e_fci - e_hf)*1000:.8f} mHa')

    # Аналіз хвильової функції
    print(f'\nАналіз хвильової функції:')

    # Вага HF конфігурації
    if spin == 0:
        # Для RHF: HF конфігурація є першим елементом
        c0_squared = ci_vec[0]**2
        print(f'Вага HF конфігурації: {c0_squared:.6f} '
              f'({c0_squared*100:.2f}%)')

    # Ентропія CI вектора (міра багатоконфігураційності)
    ci_vec_flat = ci_vec.ravel()
    ci_squared = ci_vec_flat**2
    ci_squared = ci_squared[ci_squared > 1e-10]  # відкидаємо дуже малі

    entropy = -np.sum(ci_squared * np.log(ci_squared))
    print(f'Ентропія CI вектора: {entropy:.6f}')

    if entropy < 0.1:
        print('  → Сильно одноконфігураційна система')
    elif entropy < 1.0:
        print('  → Помірна багатоконфігураційність')
    else:
        print('  → Сильна багатоконфігураційність')

    # Найважливіші конфігурації
    n_important = np.sum(ci_squared > 0.01)
    print(f'Конфігурацій з вагою > 1%: {n_important}')

    return e_hf, e_fci

# Приклади (тільки малі системи!)
fci_calculation_detailed('He', spin=0, basis='cc-pvdz')
fci_calculation_detailed('Be', spin=0, basis='cc-pvdz')
fci_calculation_detailed('Li', spin=1, basis='6-31g')
