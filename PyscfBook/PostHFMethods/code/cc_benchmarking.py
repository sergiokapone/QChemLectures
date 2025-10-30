import numpy as np
from pyscf import gto, scf, cc, ci, mp
import time

def compare_methods(mol_spec, basis='cc-pvdz'):
    """
    Комплексне порівняння пост-HF методів для заданої молекули.

    Функція виконує послідовні розрахунки HF, MP2, CISD, CCSD та CCSD(T)
    для однієї молекулярної системи та порівнює:
      - Повні та кореляційні енергії
      - Час виконання (обчислювальну складність)
      - Відхилення від "золотого стандарту" CCSD(T)
      - Кластерні амплітуди T1 і T2 (діагностика одноконфігураційності)

    Параметри
    ----------
    mol_spec : str
        Геометрія молекули у форматі PySCF (наприклад, 'H 0 0 0; F 0 0 1.1')
    basis : str, optional
        Базисний набір (за замовчуванням 'cc-pvdz')

    Повертає
    --------
    dict
        Словник з енергіями та часом виконання всіх методів

    Приклади
    --------
    >>> compare_methods('H 0 0 0; F 0 0 1.1', basis='cc-pvtz')
    """

    print("="*60)
    print(f"Молекула: {mol_spec}")
    print(f"Базис: {basis}")
    print("="*60)

    # Створення молекули
    mol = gto.M(atom=mol_spec, basis=basis, verbose=0)

    # Hartree-Fock
    mf = scf.RHF(mol).run(verbose=0)
    e_hf = mf.e_tot
    print(f"\n{'Метод':<15} {'Енергія (Ha)':<18} {'Час (с)':<10}")
    print("-"*60)
    print(f"{'HF':<15} {e_hf:<18.10f} {'---':<10}")

    # MP2
    t0 = time.time()
    mymp = mp.MP2(mf).run(verbose=0)
    t_mp2 = time.time() - t0
    e_mp2 = mymp.e_tot
    print(f"{'MP2':<15} {e_mp2:<18.10f} {t_mp2:<10.4f}")

    # CISD
    t0 = time.time()
    myci = ci.CISD(mf).run(verbose=0)
    t_cisd = time.time() - t0
    e_cisd = myci.e_tot
    print(f"{'CISD':<15} {e_cisd:<18.10f} {t_cisd:<10.4f}")

    # CCSD
    t0 = time.time()
    mycc = cc.CCSD(mf).run(verbose=0)
    t_ccsd = time.time() - t0
    e_ccsd = mycc.e_tot
    print(f"{'CCSD':<15} {e_ccsd:<18.10f} {t_ccsd:<10.4f}")

    # CCSD(T)
    t0 = time.time()
    et = mycc.ccsd_t()
    t_ccsdt = time.time() - t0
    e_ccsd_t = e_ccsd + et
    print(f"{'CCSD(T)':<15} {e_ccsd_t:<18.10f} {t_ccsdt:<10.4f}")

    # Аналіз кореляційних енергій
    print("\n" + "="*60)
    print("Аналіз кореляційних енергій (відносно HF):")
    print("-"*60)
    e_corr_mp2 = e_mp2 - e_hf
    e_corr_cisd = e_cisd - e_hf
    e_corr_ccsd = e_ccsd - e_hf
    e_corr_ccsd_t = e_ccsd_t - e_hf

    print(f"{'MP2:':<15} {e_corr_mp2:>12.6f} Ha  ({e_corr_mp2*627.5:>8.2f} kcal/mol)")
    print(f"{'CISD:':<15} {e_corr_cisd:>12.6f} Ha  ({e_corr_cisd*627.5:>8.2f} kcal/mol)")
    print(f"{'CCSD:':<15} {e_corr_ccsd:>12.6f} Ha  ({e_corr_ccsd*627.5:>8.2f} kcal/mol)")
    print(f"{'CCSD(T):':<15} {e_corr_ccsd_t:>12.6f} Ha  ({e_corr_ccsd_t*627.5:>8.2f} kcal/mol)")

    # Різниці між методами
    print("\n" + "="*60)
    print("Різниці відносно CCSD(T) (хімічна точність ~1 kcal/mol):")
    print("-"*60)
    print(f"{'MP2:':<15} {(e_mp2-e_ccsd_t)*627.5:>8.2f} kcal/mol")
    print(f"{'CISD:':<15} {(e_cisd-e_ccsd_t)*627.5:>8.2f} kcal/mol")
    print(f"{'CCSD:':<15} {(e_ccsd-e_ccsd_t)*627.5:>8.2f} kcal/mol")

    # Внесок (T)-корекції
    print(f"\nВнесок (T)-корекції: {et*627.5:.2f} kcal/mol")
    print(f"                     {et/e_corr_ccsd_t*100:.1f}% від повної кореляції")

    # Аналіз амплітуд T1 і T2
    print("\n" + "="*60)
    print("Аналіз кластерних амплітуд:")
    print("-"*60)
    t1_norm = np.linalg.norm(mycc.t1)
    t2_norm = np.linalg.norm(mycc.t2)
    print(f"Норма T1: {t1_norm:.6f}")
    print(f"Норма T2: {t2_norm:.6f}")
    print(f"Співвідношення T2/T1: {t2_norm/t1_norm:.2f}")

    max_t1 = np.max(np.abs(mycc.t1))
    max_t2 = np.max(np.abs(mycc.t2))
    print(f"\nМаксимальна амплітуда T1: {max_t1:.6f}")
    print(f"Максимальна амплітуда T2: {max_t2:.6f}")

    if max_t1 > 0.02:
        print("\n⚠ УВАГА: Велика T1-амплітуда (>0.02) може вказувати")
        print("  на багатоконфігураційний характер системи!")

    return {
        'e_hf': e_hf, 'e_mp2': e_mp2, 'e_cisd': e_cisd,
        'e_ccsd': e_ccsd, 'e_ccsd_t': e_ccsd_t,
        't_mp2': t_mp2, 't_cisd': t_cisd, 't_ccsd': t_ccsd
    }


def test_size_extensivity():
    """
    Чисельна перевірка властивості size-extensivity методів CI і CC.

    Функція демонструє критичну різницю між методами:
      - CISD (Configuration Interaction) НЕ є size-extensive
      - CCSD (Coupled Cluster) Є size-extensive

    Тест порівнює:
      E(два далекі H₂) vs 2 × E(один H₂)

    Для size-extensive методу ці величини мають збігатися (різниця ~ 0).
    Для CISD спостерігається систематична помилка масштабування.

    Це пояснює, чому CC перевершує CI для великих молекулярних систем.

    Виводить
    --------
    Таблицю порівняння енергій з обчисленою різницею у мілі-Хартрі.
    """

    print("\n" + "="*60)
    print("ТЕСТ SIZE-EXTENSIVITY")
    print("="*60)
    print("\nПорівнюємо енергію двох незалежних молекул H2")
    print("з енергією однієї молекули H2, помноженою на 2\n")

    # Одна молекула H2
    mol1 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz', verbose=0)
    mf1 = scf.RHF(mol1).run(verbose=0)

    # Дві далекі молекули H2
    mol2 = gto.M(
        atom='H 0 0 0; H 0 0 0.74; H 10 0 0; H 10 0 0.74',
        basis='cc-pvdz', verbose=0
    )
    mf2 = scf.RHF(mol2).run(verbose=0)

    print(f"{'Метод':<12} {'E(2×H2)':<16} {'2×E(H2)':<16} {'Δ (mHa)':<12}")
    print("-"*60)

    # HF (завжди size-extensive)
    e_hf_1 = mf1.e_tot
    e_hf_2 = mf2.e_tot
    print(f"{'HF':<12} {e_hf_2:<16.8f} {2*e_hf_1:<16.8f} {(e_hf_2-2*e_hf_1)*1000:<12.4f}")

    # CISD (не size-extensive!)
    myci1 = ci.CISD(mf1).run(verbose=0)
    myci2 = ci.CISD(mf2).run(verbose=0)
    e_cisd_1 = myci1.e_tot
    e_cisd_2 = myci2.e_tot
    delta_cisd = (e_cisd_2 - 2*e_cisd_1) * 1000
    print(f"{'CISD':<12} {e_cisd_2:<16.8f} {2*e_cisd_1:<16.8f} {delta_cisd:<12.4f}")

    # CCSD (size-extensive!)
    mycc1 = cc.CCSD(mf1).run(verbose=0)
    mycc2 = cc.CCSD(mf2).run(verbose=0)
    e_ccsd_1 = mycc1.e_tot
    e_ccsd_2 = mycc2.e_tot
    delta_ccsd = (e_ccsd_2 - 2*e_ccsd_1) * 1000
    print(f"{'CCSD':<12} {e_ccsd_2:<16.8f} {2*e_ccsd_1:<16.8f} {delta_ccsd:<12.4f}")

    print("\n" + "-"*60)
    print("Висновок:")
    print(f"  CISD помилка: {abs(delta_cisd):.4f} mHa (НЕ size-extensive)")
    print(f"  CCSD помилка: {abs(delta_ccsd):.4f} mHa (size-extensive ✓)")
    print("\nМетод CC коректно масштабується з розміром системи!")


def analyze_convergence():
    """
    Аналіз збіжності ієрархії квантово-хімічних методів.

    Функція показує, як кожен наступний метод у ієрархії:
      HF → MP2 → CCSD → CCSD(T)
    систематично покращує опис електронної кореляції.

    Демонструє:
      - Відсоток відновленої кореляційної енергії на кожному рівні
      - Внесок (T)-корекції (потрійних збуджень)
      - Наскільки MP2 наближається до CCSD(T)

    Це допомагає зрозуміти, чому CCSD(T) називають "золотим стандартом":
    він дає ~99% точності Full CI за прийнятну обчислювальну ціну O(N⁷).

    Виводить
    --------
    Таблицю енергій з розподілом кореляційного внеску по методах.
    """

    print("\n" + "="*60)
    print("ЗБІЖНІСТЬ ІЄРАРХІЇ МЕТОДІВ")
    print("="*60)
    print("\nПорівняння для молекули H2O\n")

    mol = gto.M(
        atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
        basis='cc-pvtz',
        verbose=0
    )

    mf = scf.RHF(mol).run(verbose=0)

    # Послідовність методів
    methods = []

    # HF
    methods.append(('HF', mf.e_tot, 0))

    # MP2
    mymp = mp.MP2(mf).run(verbose=0)
    methods.append(('MP2', mymp.e_tot, mymp.e_tot - mf.e_tot))

    # CCSD
    mycc = cc.CCSD(mf).run(verbose=0)
    methods.append(('CCSD', mycc.e_tot, mycc.e_tot - mf.e_tot))

    # CCSD(T)
    et = mycc.ccsd_t()
    methods.append(('CCSD(T)', mycc.e_tot + et, mycc.e_tot + et - mf.e_tot))

    # Вивід результатів
    print(f"{'Метод':<12} {'Повна енергія (Ha)':<20} {'E_corr (Ha)':<15} {'% від CCSD(T)':<15}")
    print("-"*70)

    e_corr_ref = methods[-1][2]  # CCSD(T)

    for name, e_tot, e_corr in methods:
        if e_corr != 0:
            percent = e_corr / e_corr_ref * 100
            print(f"{name:<12} {e_tot:<20.10f} {e_corr:<15.8f} {percent:<15.1f}")
        else:
            print(f"{name:<12} {e_tot:<20.10f} {'---':<15} {'---':<15}")

    print("\n" + "-"*70)
    print("Спостереження:")
    print(f"  MP2 відновлює {methods[1][2]/e_corr_ref*100:.1f}% кореляційної енергії")
    print(f"  CCSD додає ще {(methods[2][2]-methods[1][2])/e_corr_ref*100:.1f}%")
    print(f"  (T)-корекція додає фінальні {et/e_corr_ref*100:.1f}%")


# ============================================================
# ОСНОВНА ПРОГРАМА: cc_benchmarking.py
# ============================================================
"""
Бенчмаркінг методів зв'язаних кластерів (Coupled Cluster Benchmarking)

Ця програма виконує комплексну валідацію та порівняння методів
зв'язаних кластерів (CC) у квантовій хімії.

Основні можливості:
--------------------
1. Порівняння точності методів HF, MP2, CISD, CCSD, CCSD(T)
   - Абсолютні та кореляційні енергії
   - Відхилення від "золотого стандарту" CCSD(T)
   - Оцінка часу виконання (обчислювальної складності)

2. Тест size-extensivity (адитивності енергії)
   - Демонстрація проблеми методу CI
   - Підтвердження коректності методу CC
   - Чисельна перевірка на системі 2×H₂

3. Діагностика кластерних амплітуд T₁ і T₂
   - Перевірка одноконфігураційності системи
   - Виявлення мультиреферентного характеру
   - Аналіз домінуючих збуджень

4. Аналіз збіжності ієрархії методів
   - Розподіл кореляційної енергії по рівнях теорії
   - Внесок (T)-корекції потрійних збуджень
   - Обґрунтування статусу CCSD(T) як "золотого стандарту"

Використання:
-------------
Запустіть програму для автоматичного виконання всіх тестів:
    $ python cc_benchmarking.py

Або використовуйте окремі функції для власних молекул:
    >>> compare_methods('O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587')
    >>> test_size_extensivity()
    >>> analyze_convergence()

Вимоги:
-------
- PySCF >= 2.0
- NumPy
- Python >= 3.7

Автор: [Ваше ім'я]
Версія: 1.0
Ліцензія: MIT
"""

if __name__ == '__main__':

    # 1. Порівняння методів для HF
    print("\n" + "#"*60)
    print("# 1. ПОРІВНЯННЯ МЕТОДІВ ДЛЯ МОЛЕКУЛИ HF")
    print("#"*60)
    compare_methods('H 0 0 0; F 0 0 1.1', basis='cc-pvdz')

    # 2. Тест size-extensivity
    print("\n\n" + "#"*60)
    print("# 2. ПЕРЕВІРКА SIZE-EXTENSIVITY")
    print("#"*60)
    test_size_extensivity()

    # 3. Збіжність ієрархії методів
    print("\n\n" + "#"*60)
    print("# 3. АНАЛІЗ ЗБІЖНОСТІ")
    print("#"*60)
    analyze_convergence()

    # 4. Порівняння для складнішої системи
    print("\n\n" + "#"*60)
    print("# 4. АМІАК NH3 (СКЛАДНІША СИСТЕМА)")
    print("#"*60)
    compare_methods(
        'N 0 0 0; H 0 0.937 0.383; H 0.811 -0.469 0.383; H -0.811 -0.469 0.383',
        basis='cc-pvdz'
    )

