"""
compare_tpss_pbe.py
===================
Порівняння енергій meta-GGA (TPSS) та GGA (PBE) функціоналів
для атома заліза (Fe) у високоспіновому стані (⁵D).

- Базис: def2-tzvp
- Метод: unrestricted KS (UKS)
- Мета: продемонструвати вплив кінетичної густини τ(r) на енергію системи

Очікується, що TPSS дає нижчу енергію, ніж PBE,
завдяки кращому опису кореляції у d-електронних системах.
"""

from pyscf import gto, dft

mol = gto.M(
    atom="Fe 0 0 0",
    basis="def2-tzvp",
    spin=4,  # ⁵D основний стан
)

# ====== TPSS meta-GGA ======
mf = dft.UKS(mol)
mf.xc = "tpss"
mf.verbose = 4
energy_tpss = mf.kernel()

print(f"\nЕнергія Fe (TPSS): {energy_tpss:.8f} Ha")

# ====== PBE GGA ======
mf_pbe = dft.UKS(mol)
mf_pbe.xc = "pbe"
mf_pbe.verbose = 0
energy_pbe = mf_pbe.kernel()

print(f"Енергія Fe (PBE):  {energy_pbe:.8f} Ha")
print(f"Різниця (TPSS-PBE): {(energy_tpss - energy_pbe) * 1000:.2f} mHa")
