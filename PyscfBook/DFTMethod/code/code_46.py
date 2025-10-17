# ============================================================
# code_46.py - Конвергенція до базисної межі
# ============================================================
from pyscf import gto, dft
import numpy as np

print("Конвергенція до базисної межі для N2")
print("=" * 80)

basis_sets = [
    'sto-3g',
    '6-31g',
    '6-31g*',
    '6-311g**',
    'cc-pvdz',
    'cc-pvtz',
    'cc-pvqz',
    'aug-cc-pvdz',
    'aug-cc-pvtz',
    'aug-cc-pvqz',
]

print(f"\n{'Базисний набір':<20} {'N_AO':<10} {'Енергія (Ha)':<18} "
      f"{'ΔE (mHa)':<12} {'r_opt (Å)':<12}")
print("-" * 80)

results_basis = []

for basis in basis_sets:
    try:
        mol = gto.M(
            atom='N 0 0 0; N 0 0 1.098',
            basis=basis,
            unit='angstrom'
        )

        mf = dft.RKS(mol)
        mf.xc = 'pbe0'
        mf.verbose = 0
        energy = mf.kernel()

        n_ao = mol.nao
        results_basis.append((basis, n_ao, energy))

        # Різниця з попереднім
        if len(results_basis) > 1:
            delta = (energy - results_basis[-2][2]) * 1000  # mHa
        else:
            delta = 0.0

        print(f"{basis:<20} {n_ao:<10} {energy:<18.8f} {delta:<12.4f} {'1.098':<12}")

    except Exception as e:
        print(f"{basis:<20} {'N/A':<10} {'FAILED':<18} {'':<12} {'':<12}")

# Екстраполяція
print("\n" + "=" * 80)
print("Екстраполяція до базисної межі:")
print("Використовуємо формулу: E(l) = E_∞ + A/l³")

# Беремо cc-pVXZ серію
ccpv_results = [(b, e) for b, n, e in results_basis
                if b.startswith('cc-pv') and 'aug' not in b]

if len(ccpv_results) >= 2:
    # l=2 (DZ), l=3 (TZ), l=4 (QZ)
    l_values = np.array([2, 3, 4][:len(ccpv_results)])
    energies = np.array([e for _, e in ccpv_results])

    # Лінійна регресія E vs 1/l³
    from scipy.optimize import curve_fit

    def extrapolation(l, E_inf, A):
        return E_inf + A / l**3

    popt, _ = curve_fit(extrapolation, l_values, energies)
    E_inf, A = popt

    print(f"  E_∞ = {E_inf:.8f} Hartree")
    print(f"  A   = {A:.6f}")
    print(f"\nПорівняння з найбільшим базисом:")
    print(f"  E(cc-pVQZ) = {ccpv_results[-1][1]:.8f} Ha")
    print(f"  ΔE         = {(E_inf - ccpv_results[-1][1])*1000:.4f} mHa")
    
