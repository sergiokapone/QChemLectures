# ============================================================
# methanol_nmr_shifts.py
# Розрахунок хімічних зсувів ЯМР для метанолу
# ============================================================

from pyscf import gto, scf, dft
from pyscf.prop.nmr import rhf as nmr_rhf
from pyscf.prop.nmr import rks as nmr_rks

# Молекула метанолу
mol = gto.M(
    atom="""
    C   0.0000   0.0000   0.0000
    O   1.4200   0.0000   0.0000
    H  -0.3867   1.0272   0.0000
    H  -0.3867  -0.5136   0.8895
    H  -0.3867  -0.5136  -0.8895
    H   1.7533   0.8165   0.0000
    """,
    basis="6-31g*",  # Для ЯМР краще використовувати pcS-n базиси
    unit="angstrom",
)

print("Хімічні зсуви ЯМР для CH3OH")
print("=" * 60)

# DFT розрахунок (B3LYP зазвичай краще для ЯМР)
print("\nSCF розрахунок (B3LYP/6-31G*)...")
mf = dft.RKS(mol)
mf.xc = "b3lyp"
mf.verbose = 0
mf.kernel()

# Обчислюємо тензори екранування (GIAO метод за замовчуванням)
print("\nОбчислення ЯМР екранування (GIAO)...")
nmr = nmr_rks.NMR(mf)
shielding = nmr.kernel()

print("\nТензори магнітного екранування σ (ppm):")
print("-" * 60)

atom_labels = ["C", "O", "H(CH3)", "H(CH3)", "H(CH3)", "H(OH)"]

for ia in range(mol.natm):
    sigma = shielding[ia]
    sigma_iso = np.trace(sigma) / 3

    print(f"\n{atom_labels[ia]}:")
    print(f"  σ_iso = {sigma_iso:.2f} ppm")
    print(f"  σ_xx = {sigma[0,0]:.2f}, σ_yy = {sigma[1,1]:.2f}, σ_zz = {sigma[2,2]:.2f}")

# Конвертуємо в хімічні зсуви відносно стандартів
# Для цього потрібно розрахувати екранування стандарту (TMS)
print("\n" + "=" * 60)
print("Хімічні зсуви δ (відносно TMS):")
print("-" * 60)

# Типові σ(TMS): ¹H ≈ 32 ppm, ¹³C ≈ 182 ppm (B3LYP/6-31G*)
sigma_TMS_H = 32.0
sigma_TMS_C = 182.0

for ia in range(mol.natm):
    symbol = mol.atom_symbol(ia)
    sigma = shielding[ia]
    sigma_iso = np.trace(sigma) / 3

    if symbol == "C":
        delta = sigma_TMS_C - sigma_iso
        print(f"\n¹³C: δ = {delta:.1f} ppm")
        print(f"     (експеримент: 49.9 ppm)")
    elif symbol == "H":
        delta = sigma_TMS_H - sigma_iso
        if ia <= 4:
            print(f"\n¹H (CH₃): δ = {delta:.1f} ppm")
            if ia == 2:
                print(f"          (експеримент: 3.3 ppm)")
        else:
            print(f"\n¹H (OH): δ = {delta:.1f} ppm")
            print(f"         (експеримент: 4.8 ppm)")

print("\nПримітка:")
print("- Точність: ±2-5 ppm для ¹H, ±5-10 ppm для ¹³C")
print("- Для кращих результатів використовуйте pcS-n базиси")
print("- Врахування розчинника покращує точність")

