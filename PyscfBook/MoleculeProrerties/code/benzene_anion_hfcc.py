# ============================================================
# benzene_anion_hfcc.py
# Константи надтонкої взаємодії для аніон-радикала бензолу
# ============================================================

from pyscf import gto, scf, dft
from pyscf.prop.hfc import uhf as hfc_uhf
from pyscf.prop.hfc import uks as hfc_uks
import numpy as np

# Аніон-радикал бензолу C6H6•⁻
mol = gto.M(
    atom="""
    C   1.3970   0.0000   0.0000
    C   0.6985   1.2100   0.0000
    C  -0.6985   1.2100   0.0000
    C  -1.3970   0.0000   0.0000
    C  -0.6985  -1.2100   0.0000
    C   0.6985  -1.2100   0.0000
    H   2.4810   0.0000   0.0000
    H   1.2405   2.1486   0.0000
    H  -1.2405   2.1486   0.0000
    H  -2.4810   0.0000   0.0000
    H  -1.2405  -2.1486   0.0000
    H   1.2405  -2.1486   0.0000
    """,
    basis="6-311g**",
    charge=-1,
    spin=1,  # Дублет, один непарений електрон
    symmetry=False,
)

print("Константи НТВ для аніон-радикала бензолу C₆H₆•⁻")
print("=" * 60)

# UKS розрахунок (B3LYP краще для НТВ)
print("\nUKS розрахунок (B3LYP/6-311G**)...")
mf = dft.UKS(mol)
mf.xc = "b3lyp"
mf.verbose = 0
mf.kernel()

# Спіновий стан
s2 = mf.spin_square()[0]
print(f"\n<S²> = {s2:.6f} (очікується 0.750)")

# Обчислюємо константи НТВ
print("\nОбчислення констант надтонкої взаємодії...")
hfc = hfc_uks.HFC(mf)
a_iso, a_dip = hfc.kernel()

print("\nКонстанти НТВ:")
print("=" * 60)

# Конвертуємо з au в МГц
# 1 au = 1420.4057 МГц для протона
au2mhz_H = 1420.4057
au2mhz_C = 360.3  # Для ¹³C (менший магнітний момент)

print(f"\n{'Атом':<8} {'A_iso (МГц)':<15} {'A_dip (МГц)':<30}")
print("-" * 60)

for ia in range(mol.natm):
    symbol = mol.atom_symbol(ia)

    if symbol == "C":
        a_iso_mhz = a_iso[ia] * au2mhz_C
    else:  # H
        a_iso_mhz = a_iso[ia] * au2mhz_H

    # Анізотропна частина (тензор)
    a_dip_trace = (a_dip[ia][0,0] + a_dip[ia][1,1] + a_dip[ia][2,2])

    if symbol == "C":
        a_dip_str = f"[{a_dip[ia][2,2]*au2mhz_C:.1f}]"
    else:
        a_dip_str = f"[{a_dip[ia][2,2]*au2mhz_H:.1f}]"

    if ia < 6:  # Атоми вуглецю
        print(f"{symbol}{ia+1:<6} {a_iso_mhz:>13.2f}  {a_dip_str:>28}")
    else:  # Атоми водню
        print(f"{symbol}{ia-5:<6} {a_iso_mhz:>13.2f}  {a_dip_str:>28}")

print("\n" + "=" * 60)
print("Експериментальні значення:")
print("  A_iso(¹H) ≈ -17.5 МГц (усі H еквівалентні)")
print("  A_iso(¹³C) ≈ +12 МГц")

print("\nІнтерпретація:")
print("- Негативна A_iso для H: поляризація спіну")
print("- Позитивна A_iso для C: пряма π-електронна густина")
print("- Анізотропія мала для планарних ароматичних систем")
print("- Симетрія: всі H та всі C еквівалентні")

