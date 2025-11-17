"""
uhf_convergence_strategies.py
--------------------------------------------------------------
Покрокова стабілізація SCF для важких або погано збіжних систем.

Мета:
  • продемонструвати різні стратегії покращення збіжності SCF:
      1. стандартний UHF;
      2. level shift;
      3. збільшення DIIS-простору;
      4. атомний початковий guess;
      5. метод Ньютона–Рафсона;
      6. дробові заповнення (fractional occupations);
  • показати, як послідовне застосування цих методів
    може допомогти знайти збіжний розв’язок навіть
    для перехідних металів.

Метод:
  UHF (Unrestricted Hartree–Fock) у PySCF.
--------------------------------------------------------------
"""

from pyscf import gto, scf


def convergence_strategies(symbol, spin, basis="def2-svp"):
    """
    Покрокова стабілізація SCF для складних систем
    """
    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

    strategies = [
        ("Стандартний UHF", {}),
        ("UHF з level shift", {"level_shift": 0.3}),
        ("UHF з більшим DIIS", {"diis_space": 15}),
        ("UHF з атомним guess", {"init_guess": "atom"}),
        ("Метод Ньютона–Рафсона", {"newton": True}),
        ("Дробові заповнення", {"frac_occ": True}),
    ]

    print(f"=== Тестування стратегій конвергенції для {symbol} ===")

    for name, params in strategies:
        print(f"\n→ {name}")
        mf = scf.UHF(mol)
        mf.max_cycle = 100
        mf.conv_tol = 1e-9

        for k, v in params.items():
            if k in ("newton", "frac_occ"):
                continue
            setattr(mf, k, v)

        try:
            if params.get("newton"):
                mf = mf.newton()
            if params.get("frac_occ"):
                mf = scf.addons.frac_occ(mf)
            energy = mf.kernel()
            if mf.converged:
                print(f"✓ Успіх: E = {energy:.8f} Ha")
                return mf, energy
            else:
                print("✗ Не конвергувало")
        except Exception as e:
            print(f"✗ Помилка: {e}")
    print("\n⚠ Жодна стратегія не спрацювала.")
    return None, None


# Приклад: перехідний метал
mf, e = convergence_strategies("Co", spin=3)
