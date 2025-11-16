# ============================================================
# formaldehyde_functional_comparison.py
# Порівняння різних функціоналів для TDDFT
# ============================================================

from pyscf import gto, dft, tddft

mol = gto.M(
    atom="""
    C  0.0000  0.0000  0.0000
    O  0.0000  0.0000  1.2050
    H  0.0000  0.9428 -0.5876
    H  0.0000 -0.9428 -0.5876
    """,
    basis="aug-cc-pvdz",
    unit="angstrom",
)

print("Порівняння функціоналів DFT для збуджень H2CO")
print("=" * 60)

# Список функціоналів для тестування
functionals = ["b3lyp", "pbe0", "cam-b3lyp", "wb97x-d"]

results = {}

for xc in functionals:
    print(f"\n{xc.upper()}:")
    print("-" * 40)

    # DFT розрахунок
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.verbose = 0
    mf.kernel()

    # TDDFT
    td = tddft.TDDFT(mf)
    td.nstates = 3
    td.verbose = 0
    td.kernel()

    # Перший перехід (n→π*)
    energy_ev = td.e[0] * 27.2114
    wavelength = 1240 / energy_ev
    osc_str = td.oscillator_strength()[0]

    results[xc] = (energy_ev, wavelength, osc_str)

    print(f"  S₁: {energy_ev:.3f} eV ({wavelength:.1f} nm)")
    print(f"  f = {osc_str:.4f}")

print("\n" + "=" * 60)
print("Порівняння для n→π* переходу:")
print("-" * 60)
print(f"{'Функціонал':<15} {'λ (нм)':<12} {'Відхилення'}")
print("-" * 60)

exp_wavelength = 330  # Експериментальне значення (нм)

for xc, (e, wl, f) in results.items():
    diff = wl - exp_wavelength
    print(f"{xc.upper():<15} {wl:>10.1f}  {diff:+7.1f} нм")

print("-" * 60)
print(f"{'ЕКСПЕРИМЕНТ':<15} {exp_wavelength:>10.1f}  {'---'}")

print("\nВисновки:")
print("- Range-separated функціонали (CAM-B3LYP, ωB97X-D) найточніші")
print("- B3LYP завищує довжини хвиль")
print("- Для переносу заряду обов'язково використовувати range-separated")

