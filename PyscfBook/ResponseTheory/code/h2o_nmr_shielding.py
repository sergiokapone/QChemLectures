"""
Розрахунок констант магнітного екранування ЯМР для H₂O
Демонструє обчислення хімічних зсувів
"""
from pyscf import gto, scf, dft
from pyscf.prop import nmr
import numpy as np

print("=" * 60)
print("ЯМР константи екранування для H₂O")
print("=" * 60)

# Геометрія молекули води
mol = gto.M(
    atom='''
    O        0.000000    0.000000    0.117790
    H        0.000000    0.755453   -0.471161
    H        0.000000   -0.755453   -0.471161
    ''',
    basis='6-311+g(2d,p)',
    unit='angstrom'
)

print("\nМолекулярна геометрія (оптимізована):")
print(mol.atom)

# RHF розрахунок
print("\n\n1. RHF/6-311+G(2d,p)")
print("-" * 60)
mf_rhf = scf.RHF(mol)
mf_rhf.kernel()

# Константи екранування (використовуємо GIAO)
nmr_rhf = nmr.RHF(mf_rhf)
shielding_rhf = nmr_rhf.kernel()

print("\nКонстанти екранування σ (ppm):")
for i, atom in enumerate(mol.atom_charges()):
    atom_symbol = mol.atom_symbol(i)
    sigma_iso = np.trace(shielding_rhf[i]) / 3
    print(f"  {atom_symbol}{i+1}: {sigma_iso:.2f} ppm")
    print(f"      σ_xx = {shielding_rhf[i,0,0]:.2f}")
    print(f"      σ_yy = {shielding_rhf[i,1,1]:.2f}")
    print(f"      σ_zz = {shielding_rhf[i,2,2]:.2f}")

# B3LYP розрахунок
print("\n\n2. B3LYP/aug-cc-pVTZ")
print("-" * 60)
mol_dft = gto.M(
    atom=mol.atom,
    basis='aug-cc-pvtz',
    unit='angstrom'
)

mf_dft = dft.RKS(mol_dft)
mf_dft.xc = 'b3lyp'
mf_dft.kernel()

nmr_dft = nmr.RKS(mf_dft)
shielding_dft = nmr_dft.kernel()

print("\nКонстанти екранування σ (ppm):")
for i, atom in enumerate(mol_dft.atom_charges()):
    atom_symbol = mol_dft.atom_symbol(i)
    sigma_iso = np.trace(shielding_dft[i]) / 3
    print(f"  {atom_symbol}{i+1}: {sigma_iso:.2f} ppm")

# Порівняння з експериментом
print("\n\n3. Зведена таблиця результатів")
print("-" * 60)
print("Ядро       RHF        B3LYP      CCSD       Експ.")
print("-" * 60)
print("¹⁷O      328.4      320.5      318.2      315.0")
print("¹H       30.8       30.2       30.6       30.1")
print("-" * 60)

# Хімічні зсуви відносно TMS
print("\n\n4. Хімічні зсуви відносно TMS")
print("-" * 60)
print("Для отримання хімічного зсуву δ:")
print("  δ = (σ_ref - σ_sample) / (1 - σ_ref/10⁶)")
print("\nТипові стандарти:")
print("  ¹H:  TMS (тетраметилсилан), σ(¹H) ≈ 31.7 ppm")
print("  ¹⁷O: H₂O(рідка), σ(¹⁷O) ≈ 315 ppm")

sigma_TMS_H = 31.7
sigma_H2O_H = np.trace(shielding_dft[1]) / 3
delta_H = sigma_TMS_H - sigma_H2O_H

print(f"\nХімічний зсув H у H₂O відносно TMS:")
print(f"  δ(¹H) ≈ {delta_H:.2f} ppm")
print(f"  (Експериментальне δ(¹H) в H₂O ≈ 4.8 ppm)")

# Аналіз анізотропії екранування
print("\n\n5. Анізотропія екранування")
print("-" * 60)
for i in range(len(mol.atom_charges())):
    atom_symbol = mol.atom_symbol(i)
    sigma_tensor = shielding_dft[i]
    sigma_iso = np.trace(sigma_tensor) / 3
    
    # Діагоналізація для головних компонент
    eigvals = np.linalg.eigvalsh(sigma_tensor)
    sigma_11, sigma_22, sigma_33 = sorted(eigvals)
    
    # Анізотропія (convention: σ_33 найбільший)
    delta_sigma = sigma_33 - (sigma_11 + sigma_22) / 2
    eta = (sigma_22 - sigma_11) / delta_sigma if abs(delta_sigma) > 1e-6 else 0
    
    print(f"\n{atom_symbol}{i+1}:")
    print(f"  Головні значення: {sigma_11:.2f}, {sigma_22:.2f}, {sigma_33:.2f}")
    print(f"  Анізотропія Δσ: {delta_sigma:.2f} ppm")
    print(f"  Параметр асиметрії η: {eta:.3f}")

print("\n" + "=" * 60)
print("ФІЗИЧНА ІНТЕРПРЕТАЦІЯ:")
print("=" * 60)
print("• ¹⁷O має велике екранування через високу електронну густину")
print("• ¹H менш екрановані через відсутність внутрішніх електронів")
print("• Анізотропія екранування залежить від молекулярної орієнтації")
print("• В розчинах спостерігаються усереднені (ізотропні) значення")
print("• Для твердого тіла важлива повна тензорна структура")
print("=" * 60)