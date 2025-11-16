#!/usr/bin/env python3
"""
Порівняння парамагнітних молекул NO та O2
Аналіз спінових станів, магнітних моментів та електронної структури
"""

from pyscf import gto, scf, dft
import numpy as np
import matplotlib.pyplot as plt

def calculate_magnetic_properties(molecule, distance, spin, method='UHF', 
                                  xc=None, basis='cc-pvdz'):
    """
    Розрахунок магнітних властивостей молекули
    
    Parameters:
    -----------
    molecule : str
        'NO' або 'O2'
    distance : float
        Міжядерна відстань (Å)
    spin : int
        2S (кількість неспарених електронів)
    method : str
        'UHF' або 'UKS'
    xc : str
        Функціонал для DFT
    basis : str
        Базисний набір
    """
    
    if molecule == 'NO':
        atom_str = f'N 0 0 0; O 0 0 {distance}'
    elif molecule == 'O2':
        atom_str = f'O 0 0 0; O 0 0 {distance}'
    else:
        raise ValueError("Molecule must be 'NO' or 'O2'")
    
    mol = gto.M(
        atom=atom_str,
        basis=basis,
        spin=spin,
        symmetry=True,
        verbose=0
    )
    
    # Вибір методу
    if method == 'UHF':
        mf = scf.UHF(mol)
    elif method == 'UKS':
        mf = dft.UKS(mol)
        mf.xc = xc if xc else 'pbe'
    else:
        raise ValueError("Method must be 'UHF' or 'UKS'")
    
    mf.verbose = 0
    mf.conv_tol = 1e-10
    energy = mf.kernel()
    
    if not mf.converged:
        return None
    
    # Розрахунок магнітних властивостей
    s2 = mf.spin_square()[0]
    expected_s2 = spin * (spin + 2) / 4
    
    # Магнітний момент (в магнетонах Бора)
    # μ ≈ √(S(S+1)) * g_e ≈ √(S(S+1)) * 2 μ_B
    S = spin / 2
    magnetic_moment = np.sqrt(S * (S + 1)) * 2  # в μ_B
    
    # Заселеності Малікена
    pop = mf.mulliken_pop(verbose=0)
    
    # Спінова густина на атомах
    if spin > 0:
        dm_alpha, dm_beta = mf.make_rdm1()
        spin_density = mf.mulliken_pop(verbose=0)[1]  # спінова заселеність
    else:
        spin_density = [0, 0]
    
    return {
        'energy': energy,
        's2': s2,
        'expected_s2': expected_s2,
        'magnetic_moment': magnetic_moment,
        'spin_density': spin_density,
        'converged': mf.converged
    }


def compare_no_o2_ground_states():
    """
    Порівняння основних станів NO та O2
    """
    
    print('='*80)
    print('ПОРІВНЯННЯ ПАРАМАГНІТНИХ МОЛЕКУЛ NO ТА O2')
    print('='*80)
    
    # Експериментальні дані
    molecules = {
        'NO': {
            'distance': 1.151,  # Å
            'spin': 1,          # дублет (1 неспарений електрон)
            'state': '²Π',
            'electrons': 15
        },
        'O2': {
            'distance': 1.208,  # Å
            'spin': 2,          # триплет (2 неспарені електрони)
            'state': '³Σ',
            'electrons': 16
        }
    }
    
    methods = ['UHF', 'B3LYP', 'PBE0']
    basis = 'cc-pvtz'
    
    results = {}
    
    for mol_name, mol_data in molecules.items():
        print(f'\n{"─"*80}')
        print(f'Молекула: {mol_name}')
        print(f'Основний стан: {mol_data["state"]}')
        print(f'Електронів: {mol_data["electrons"]}')
        print(f'Неспарених електронів: {mol_data["spin"]}')
        print(f'Міжядерна відстань: {mol_data["distance"]} Å')
        print(f'{"─"*80}')
        
        results[mol_name] = {}
        
        for method in methods:
            if method == 'UHF':
                xc = None
                method_type = 'UHF'
            else:
                xc = method.lower()
                method_type = 'UKS'
            
            print(f'\n  Метод: {method}')
            
            res = calculate_magnetic_properties(
                mol_name, 
                mol_data['distance'], 
                mol_data['spin'],
                method=method_type,
                xc=xc,
                basis=basis
            )
            
            if res:
                results[mol_name][method] = res
                
                print(f'    Енергія: {res["energy"]:.10f} Ha')
                print(f'    <S²>: {res["s2"]:.6f} (очікується {res["expected_s2"]:.6f})')
                print(f'    Забруднення спіном: {res["s2"] - res["expected_s2"]:.6f}')
                print(f'    Магнітний момент: {res["magnetic_moment"]:.4f} μ_B')
            else:
                print(f'    ✗ Не конвергувало')
    
    # Порівняльна таблиця
    print('\n' + '='*80)
    print('ПОРІВНЯЛЬНА ТАБЛИЦЯ')
    print('='*80)
    print(f'{"Властивість":30s} {"NO":20s} {"O2":20s}')
    print('─'*80)
    
    # Використовуємо B3LYP для порівняння
    if 'B3LYP' in results['NO'] and 'B3LYP' in results['O2']:
        no_res = results['NO']['B3LYP']
        o2_res = results['O2']['B3LYP']
        
        print(f'{"Неспарених електронів":30s} {molecules["NO"]["spin"]:20d} '
              f'{molecules["O2"]["spin"]:20d}')
        print(f'{"Основний стан":30s} {molecules["NO"]["state"]:20s} '
              f'{molecules["O2"]["state"]:20s}')
        print(f'{"<S²> (B3LYP)":30s} {no_res["s2"]:20.4f} {o2_res["s2"]:20.4f}')
        print(f'{"Магнітний момент (μ_B)":30s} {no_res["magnetic_moment"]:20.4f} '
              f'{o2_res["magnetic_moment"]:20.4f}')
        print(f'{"Енергія (B3LYP, Ha)":30s} {no_res["energy"]:20.8f} '
              f'{o2_res["energy"]:20.8f}')
    
    print('='*80)
    
    # Графік
    plot_magnetic_comparison(results, molecules, methods)
    
    return results


def plot_magnetic_comparison(results, molecules, methods):
    """
    Візуалізація порівняння NO та O2
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    mol_names = ['NO', 'O2']
    x = np.arange(len(mol_names))
    width = 0.25
    colors = ['blue', 'red', 'green']
    
    # 1. Порівняння енергій
    for i, (method, color) in enumerate(zip(methods, colors)):
        energies = []
        for mol in mol_names:
            if method in results[mol]:
                energies.append(results[mol][method]['energy'])
            else:
                energies.append(np.nan)
        
        ax1.bar(x + i*width, energies, width, label=method, 
               color=color, alpha=0.7)
    
    ax1.set_ylabel('Енергія (Ha)', fontsize=11)
    ax1.set_title('Повні енергії', fontsize=12, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(mol_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Порівняння <S²>
    for i, (method, color) in enumerate(zip(methods, colors)):
        s2_values = []
        for mol in mol_names:
            if method in results[mol]:
                s2_values.append(results[mol][method]['s2'])
            else:
                s2_values.append(np.nan)
        
        ax2.bar(x + i*width, s2_values, width, label=method,
               color=color, alpha=0.7)
    
    # Теоретичні значення
    expected = [molecules[mol]['spin'] * (molecules[mol]['spin'] + 2) / 4 
                for mol in mol_names]
    ax2.plot(x + width, expected, 'k--', linewidth=2, 
            marker='o', markersize=8, label='Теоретичне')
    
    ax2.set_ylabel('<S²>', fontsize=11)
    ax2.set_title('Значення <S²>', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(mol_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Магнітний момент
    for i, (method, color) in enumerate(zip(methods, colors)):
        moments = []
        for mol in mol_names:
            if method in results[mol]:
                moments.append(results[mol][method]['magnetic_moment'])
            else:
                moments.append(np.nan)
        
        ax3.bar(x + i*width, moments, width, label=method,
               color=color, alpha=0.7)
    
    ax3.set_ylabel('Магнітний момент (μ_B)', fontsize=11)
    ax3.set_title('Магнітні моменти', fontsize=12, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(mol_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Забруднення спіном
    for i, (method, color) in enumerate(zip(methods, colors)):
        contamination = []
        for mol in mol_names:
            if method in results[mol]:
                res = results[mol][method]
                cont = res['s2'] - res['expected_s2']
                contamination.append(cont)
            else:
                contamination.append(np.nan)
        
        ax4.bar(x + i*width, contamination, width, label=method,
               color=color, alpha=0.7)
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_ylabel('Забруднення спіном', fontsize=11)
    ax4.set_title('Забруднення спіном (<S²> - очікуване)', 
                 fontsize=12, fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(mol_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('no_o2_magnetic_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('\n✓ Графік збережено: no_o2_magnetic_comparison.pdf')


def spin_density_analysis():
    """
    Аналіз спінової густини для NO та O2
    """
    
    print('\n' + '='*80)
    print('АНАЛІЗ СПІНОВОЇ ГУСТИНИ')
    print('='*80)
    
    molecules = {
        'NO': (1.151, 1),
        'O2': (1.208, 2)
    }
    
    for mol_name, (distance, spin) in molecules.items():
        print(f'\n{mol_name}:')
        
        if mol_name == 'NO':
            atom_str = f'N 0 0 0; O 0 0 {distance}'
        else:
            atom_str = f'O 0 0 0; O 0 0 {distance}'
        
        mol = gto.M(
            atom=atom_str,
            basis='cc-pvdz',
            spin=spin,
            verbose=0
        )
        
        mf = dft.UKS(mol)
        mf.xc = 'b3lyp'
        mf.verbose = 0
        energy = mf.kernel()
        
        # Заселеності Малікена
        pop, chg = mf.mulliken_pop(verbose=0)
        
        # Спінова заселеність
        dm_alpha, dm_beta = mf.make_rdm1()
        s = mol.intor('int1e_ovlp')
        
        spin_pop = []
        atoms = [mol.atom_symbol(i) for i in range(mol.natm)]
        
        print(f'\n  Спінова заселеність на атомах:')
        
        # Простий аналіз (по базисних функціях кожного атома)
        ao_labels = mol.ao_labels(fmt=False)
        
        for atom_id in range(mol.natm):
            atom_symbol = mol.atom_symbol(atom_id)
            
            # Знаходимо базисні функції цього атома
            atom_aos = [i for i, label in enumerate(ao_labels) 
                       if label[0] == atom_id]
            
            # Спінова густина на атомі
            spin_dens = 0
            for i in atom_aos:
                spin_dens += (dm_alpha @ s)[i, i] - (dm_beta @ s)[i, i]
            
            print(f'    {atom_symbol}: {spin_dens:8.4f}')


def experimental_comparison():
    """
    Порівняння з експериментальними даними
    """
    
    print('\n' + '='*80)
    print('ПОРІВНЯННЯ З ЕКСПЕРИМЕНТОМ')
    print('='*80)
    
    exp_data = {
        'NO': {
            'bond_length': 1.151,  # Å
            'bond_energy': 6.49,    # eV
            'magnetic_moment': 1.73,  # μ_B (експ.)
            'state': '²Π'
        },
        'O2': {
            'bond_length': 1.208,  # Å
            'bond_energy': 5.12,    # eV
            'magnetic_moment': 2.83,  # μ_B (експ.)
            'state': '³Σ'
        }
    }
    
    print(f'\n{"Властивість":25s} {"NO":15s} {"O2":15s}')
    print('─'*80)
    print(f'{"Довжина зв\'язку (Å)":25s} {exp_data["NO"]["bond_length"]:15.3f} '
          f'{exp_data["O2"]["bond_length"]:15.3f}')
    print(f'{"Енергія зв\'язку (eV)":25s} {exp_data["NO"]["bond_energy"]:15.2f} '
          f'{exp_data["O2"]["bond_energy"]:15.2f}')
    print(f'{"Магнітний момент (μ_B)":25s} {exp_data["NO"]["magnetic_moment"]:15.2f} '
          f'{exp_data["O2"]["magnetic_moment"]:15.2f}')
    print(f'{"Основний стан":25s} {exp_data["NO"]["state"]:15s} '
          f'{exp_data["O2"]["state"]:15s}')
    print('='*80)


def main():
    """
    Головна функція
    """
    
    print('\n' + '█'*80)
    print('█' + ' '*78 + '█')
    print('█' + 'ПОРІВНЯННЯ ПАРАМАГНІТНИХ МОЛЕКУЛ NO ТА O2'.center(78) + '█')
    print('█' + ' '*78 + '█')
    print('█'*80)
    
    # 1. Порівняння основних станів
    results = compare_no_o2_ground_states()
    
    # 2. Аналіз спінової густини
    spin_density_analysis()
    
    # 3. Порівняння з експериментом
    experimental_comparison()
    
    # Висновки
    print('\n' + '='*80)
    print('ВИСНОВКИ')
    print('='*80)
    print("""
1. NO (оксид азоту):
   - Основний стан: дублет (²Π) з 1 неспареним електроном
   - Магнітний момент: ~1.73 μ_B
   - Слабко парамагнітний
   
2. O2 (молекулярний кисень):
   - Основний стан: триплет (³Σ) з 2 неспареними електронами
   - Магнітний момент: ~2.83 μ_B
   - Сильно парамагнітний
   
3. Методи розрахунку:
   - UHF: якісно правильні результати, але є забруднення спіном
   - B3LYP/PBE0: кращі енергії та менше забруднення спіном
   - Для точних магнітних властивостей рекомендується CASSCF
   
4. Обидві молекули демонструють парамагнетизм через неспарені електрони
   в антизв'язуючих π* орбіталях
    """)
    print('='*80)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Порівняння магнітних властивостей молекули O2
Розрахунок різних спінових станів та аналіз <S²>
"""

from pyscf import gto, scf, dft
import numpy as np
import matplotlib.pyplot as plt

def o2_spin_states_comparison():
    """
    Порівняння різних спінових станів O2
    Основний стан: триплет (²S = 2)
    """
    
    # Відстань O-O (експериментальна)
    bond_length = 1.208  # Angstrom
    
    print('='*70)
    print('Порівняння спінових станів молекули O2')
    print(f'Міжядерна відстань: {bond_length} Å')
    print('='*70)
    
    # Спінові стани для тестування: синглет, триплет, квінтет
    spin_states = [
        (0, 'Синглет', '¹Σ'),
        (2, 'Триплет', '³Σ'),
        (4, 'Квінтет', '⁵Σ')
    ]
    
    methods = {
        'RHF/UHF': None,
        'PBE': 'pbe',
        'B3LYP': 'b3lyp',
        'PBE0': 'pbe0'
    }
    
    results = {}
    
    for method_name, xc in methods.items():
        print(f'\n--- Метод: {method_name} ---')
        results[method_name] = {}
        
        for spin, state_name, term in spin_states:
            mol = gto.M(
                atom=f'O 0 0 0; O 0 0 {bond_length}',
                basis='cc-pvtz',
                spin=spin,
                symmetry=True,
                verbose=0
            )
            
            # Вибір методу
            if method_name == 'RHF/UHF':
                if spin == 0:
                    mf = scf.RHF(mol)
                else:
                    mf = scf.UHF(mol)
            else:
                if spin == 0:
                    mf = dft.RKS(mol)
                else:
                    mf = dft.UKS(mol)
                mf.xc = xc
            
            mf.verbose = 0
            mf.conv_tol = 1e-10
            
            try:
                energy = mf.kernel()
                
                if mf.converged:
                    # Обчислення <S²>
                    if spin == 0:
                        s2 = 0.0
                    else:
                        s2 = mf.spin_square()[0]
                    
                    expected_s2 = spin * (spin + 2) / 4
                    
                    results[method_name][spin] = {
                        'energy': energy,
                        's2': s2,
                        'expected_s2': expected_s2,
                        'state': state_name,
                        'term': term
                    }
                    
                    print(f'{state_name:10s} ({term}): E = {energy:.8f} Ha, '
                          f'<S²> = {s2:.4f} (очік. {expected_s2:.4f})')
                else:
                    print(f'{state_name:10s}: Не конвергувало')
            
            except Exception as e:
                print(f'{state_name:10s}: Помилка - {str(e)}')
    
    # Аналіз результатів
    print('\n' + '='*70)
    print('АНАЛІЗ РЕЗУЛЬТАТІВ')
    print('='*70)
    
    for method_name in methods.keys():
        if method_name not in results or not results[method_name]:
            continue
        
        print(f'\n{method_name}:')
        
        # Знаходження основного стану (найнижча енергія)
        energies = {spin: data['energy'] 
                   for spin, data in results[method_name].items()}
        
        if energies:
            ground_spin = min(energies, key=energies.get)
            ground_state = results[method_name][ground_spin]['state']
            
            print(f'  Основний стан: {ground_state}')
            
            # Енергії відносно основного стану
            e_ground = energies[ground_spin]
            
            print(f'  Відносні енергії (kcal/mol):')
            for spin, data in results[method_name].items():
                rel_e = (data['energy'] - e_ground) * 627.509
                print(f'    {data["state"]:10s}: {rel_e:8.2f}')
    
    # Графік
    plot_o2_results(results, methods)
    
    return results


def plot_o2_results(results, methods):
    """
    Візуалізація результатів для O2
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    spin_labels = {0: 'Синглет', 2: 'Триплет', 4: 'Квінтет'}
    colors = ['blue', 'red', 'green', 'orange']
    
    # Графік 1: Абсолютні енергії
    width = 0.2
    x = np.arange(len(spin_labels))
    
    for i, (method_name, color) in enumerate(zip(methods.keys(), colors)):
        if method_name not in results or not results[method_name]:
            continue
        
        energies = []
        spins_plot = []
        
        for spin in [0, 2, 4]:
            if spin in results[method_name]:
                energies.append(results[method_name][spin]['energy'])
                spins_plot.append(spin)
        
        if energies:
            x_pos = np.arange(len(spins_plot)) + i * width
            ax1.bar(x_pos, energies, width, label=method_name, 
                   color=color, alpha=0.7)
    
    ax1.set_xlabel('Спіновий стан', fontsize=12)
    ax1.set_ylabel('Енергія (Ha)', fontsize=12)
    ax1.set_title('Енергії різних станів O₂', fontsize=14)
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels([spin_labels[s] for s in [0, 2, 4]])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Графік 2: <S²> значення
    for i, (method_name, color) in enumerate(zip(methods.keys(), colors)):
        if method_name not in results or not results[method_name]:
            continue
        
        spins = []
        s2_values = []
        
        for spin in [0, 2, 4]:
            if spin in results[method_name]:
                spins.append(spin)
                s2_values.append(results[method_name][spin]['s2'])
        
        if spins:
            ax2.plot(spins, s2_values, 'o-', label=method_name, 
                    color=color, linewidth=2, markersize=8)
    
    # Теоретичні значення
    spin_theory = [0, 2, 4]
    s2_theory = [s * (s + 2) / 4 for s in spin_theory]
    ax2.plot(spin_theory, s2_theory, 'k--', linewidth=2, 
            label='Теоретичне', alpha=0.5)
    
    ax2.set_xlabel('2S', fontsize=12)
    ax2.set_ylabel('<S²>', fontsize=12)
    ax2.set_title('Забруднення спіном O₂', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('o2_magnetic_comparison.pdf', dpi=300)
    plt.show()
    
    print('\nГрафік збережено як o2_magnetic_comparison.pdf')


def o2_restricted_vs_unrestricted():
    """
    Порівняння RHF vs UHF для синглетного стану O2
    """
    
    print('\n' + '='*70)
    print('Порівняння RHF vs UHF для синглетного O2')
    print('='*70)
    
    bond_length = 1.208
    
    mol = gto.M(
        atom=f'O 0 0 0; O 0 0 {bond_length}',
        basis='cc-pvdz',
        spin=0,
        symmetry=True,
        verbose=0
    )
    
    # RHF
    print('\nRHF (restricted):')
    mf_rhf = scf.RHF(mol)
    mf_rhf.verbose = 4
    e_rhf = mf_rhf.kernel()
    print(f'E(RHF) = {e_rhf:.10f} Ha')
    
    # UHF
    print('\nUHF (unrestricted):')
    mf_uhf = scf.UHF(mol)
    mf_uhf.verbose = 4
    e_uhf = mf_uhf.kernel()
    
    s2_uhf = mf_uhf.spin_square()
    print(f'E(UHF) = {e_uhf:.10f} Ha')
    print(f'<S²> (UHF) = {s2_uhf[0]:.6f} (очікується 0.0)')
    
    # Порівняння
    print(f'\nРізниця E(UHF) - E(RHF) = {(e_uhf - e_rhf)*1000:.4f} mHa')
    print(f'Забруднення спіном: {s2_uhf[0]:.6f}')
    
    if abs(s2_uhf[0]) > 0.01:
        print('\n⚠ UHF має значне забруднення спіном!')
        print('   Синглетний стан O2 має багатоконфігураційний характер.')
        print('   Рекомендується CASSCF або MRCI.')


def o2_bond_dissociation():
    """
    Крива дисоціації O2 (триплетний стан)
    """
    
    print('\n' + '='*70)
    print('Крива дисоціації O2 (триплетний стан)')
    print('='*70)
    
    # Відстані для розрахунку
    distances = np.linspace(0.9, 4.0, 20)  # Angstrom
    
    methods = {
        'UHF': None,
        'PBE': 'pbe',
        'B3LYP': 'b3lyp'
    }
    
    results = {method: [] for method in methods}
    
    for d in distances:
        print(f'\nВідстань: {d:.2f} Å')
        
        for method_name, xc in methods.items():
            mol = gto.M(
                atom=f'O 0 0 0; O 0 0 {d}',
                basis='cc-pvdz',
                spin=2,  # Триплет
                verbose=0
            )
            
            if method_name == 'UHF':
                mf = scf.UHF(mol)
            else:
                mf = dft.UKS(mol)
                mf.xc = xc
            
            mf.verbose = 0
            mf.conv_tol = 1e-9
            
            try:
                energy = mf.kernel()
                results[method_name].append(energy)
                print(f'  {method_name:8s}: {energy:.8f} Ha')
            except:
                results[method_name].append(np.nan)
                print(f'  {method_name:8s}: не конвергувало')
    
    # Графік
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'red', 'green']
    
    for (method_name, energies), color in zip(results.items(), colors):
        # Конвертуємо в kcal/mol відносно мінімуму
        energies_arr = np.array(energies)
        valid_mask = ~np.isnan(energies_arr)
        
        if valid_mask.any():
            e_min = np.min(energies_arr[valid_mask])
            rel_energies = (energies_arr - e_min) * 627.509  # kcal/mol
            
            plt.plot(distances[valid_mask], rel_energies[valid_mask], 
                    'o-', label=method_name, color=color, 
                    linewidth=2, markersize=6)
    
    plt.xlabel('Міжядерна відстань (Å)', fontsize=12)
    plt.ylabel('Відносна енергія (kcal/mol)', fontsize=12)
    plt.title('Крива дисоціації O₂ (³Σ)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('o2_dissociation_curve.pdf', dpi=300)
    plt.show()
    
    print('\nГрафік збережено як o2_dissociation_curve.pdf')


def main():
    """
    Головна функція
    """
    
    print('\n' + '='*70)
    print('АНАЛІЗ МАГНІТНИХ ВЛАСТИВОСТЕЙ O2')
    print('='*70)
    print('\nМолекула кисню O2 є парамагнітною з двома неспареними')
    print('електронами у основному стані (³Σ).')
    print('='*70)
    
    # 1. Порівняння спінових станів
    results = o2_spin_states_comparison()
    
    # 2. RHF vs UHF для синглету
    o2_restricted_vs_unrestricted()
    
    # 3. Крива дисоціації
    o2_bond_dissociation()
    
    print('\n' + '='*70)
    print('ВИСНОВКИ')
    print('='*70)
    print('1. Основний стан O2 - триплет (³Σ) з двома неспареними електронами')
    print('2. UHF правильно описує парамагнітну природу O2')
    print('3. Синглетний стан має багатоконфігураційний характер')
    print('4. DFT (B3LYP, PBE0) дають кращі енергії порівняно з HF')
    print('5. Для точних розрахунків рекомендується CASSCF або CCSD')
    print('='*70)


if __name__ == '__main__':
    main()
