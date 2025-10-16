from pyscf import gto, scf

# --- 1. Опис атома гелію ---
mol = gto.Mole()
mol.atom  = 'He 0 0 0'
mol.basis = 'cc-pVTZ'
mol.build()

# --- 2. Розрахунок RHF ---
mf = scf.RHF(mol)
E_tot = mf.kernel()

# --- 3. Матриця густини та інтеграли ---
dm = mf.make_rdm1()
T_int = mol.intor('int1e_kin')       # оператор кінетичної енергії
V_nuc_int = mol.intor('int1e_nuc')   # оператор потенціальної енергії ядро–електрон
vhf = mf.get_veff(mol, dm)           # ефективний потенціал (Кулон + обмін)

# --- 4. Енергетичні складові ---
E_kin = (dm * T_int).sum()           # <T>
E_nuc = (dm * V_nuc_int).sum()       # <V_nuc>
E_ee  = 0.5 * (dm * vhf).sum()       # <V_ee> (Кулон + обмін)
E_tot_check = E_kin + E_nuc + E_ee   # перевірка

# --- 5. Віральне співвідношення ---
V_total = E_nuc + E_ee
virial_ratio = V_total / E_kin

# --- 6. Вивід у вигляді таблиці ---
print("\n" + "="*60)
print(" "*15 + "ЕНЕРГЕТИЧНИЙ АНАЛІЗ АТОМА He")
print("="*60)
print(f"{'Компонента':<30} {'Значення (Ha)':>20}")
print("-"*60)
print(f"{'Кінетична енергія (T)':<30} {E_kin:>20.6f}")
print(f"{'Ядро–електрони (V_nuc)':<30} {E_nuc:>20.6f}")
print(f"{'Електрон–електрон (V_ee)':<30} {E_ee:>20.6f}")
print("-"*60)
print(f"{'Повна енергія (E_tot)':<30} {E_tot:>20.6f}")
print(f"{'Перевірка суми (E_sum)':<30} {E_tot_check:>20.6f}")
print("="*60)
print(f"\n{'Віральне співвідношення':<30} {'V/T':>20}")
print("-"*60)
print(f"{'Розраховане значення':<30} {virial_ratio:>20.3f}")
print(f"{'Теоретичне значення':<30} {-2.000:>20.3f}")
print("="*60 + "\n")
