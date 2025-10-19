# ============================================================
# h2o_thermochemistry.py
# Термохімічний аналіз молекули H2O (PySCF)
# ============================================================

import numpy
from pyscf import gto, hessian
from pyscf.hessian.thermo import *
from pyscf.data import nist

# -------------------------------------------------------
# МОЛЕКУЛА ВОДИ
# -------------------------------------------------------
mol = gto.Mole()
mol.atom = '''
O  0.000000   0.000000   0.000000
H  0.000000   0.757000   0.587000
H  0.000000  -0.757000   0.587000
'''
mol.basis = '6-31g(d)'
mol.build()

mass = mol.atom_mass_list(isotope_avg=True)

# -------------------------------------------------------
# Хартрі–Фок, гессіан, термохімія
# -------------------------------------------------------
mf = scf.RHF(mol).run(verbose=0)
hess = hessian.RHF(mf).kernel()

# ---------------------------------------------------------------
#  Гармонічний аналіз коливань:
#  з гесіану обчислюються власні частоти (в см⁻¹) та нормальні моди.
#  Видаляються поступальні й обертальні ступені свободи (6 для нелінійних систем).
#  Результат містить частоти, мас-зважений гессіан і нормальні координати.
# ---------------------------------------------------------------
results = harmonic_analysis(mol, hess)
# dump_normal_mode(mol, results)   # при потребі

# ---------------------------------------------------------------
#  Розрахунок термохімічних параметрів при заданій температурі і тиску.
#  Функція thermo():
#    – використовує вібраційні частоти (results['freq_au'])
#    – враховує поступальні, обертальні та коливальні ступені свободи
#    – обчислює нульову коливальну енергію (ZPE),
#      термічні поправки до енергії, ентальпію, ентропію,
#      теплоємності (Cv, Cp) та вільну енергію Гіббса (G)
# ---------------------------------------------------------------
results = thermo(mf, results['freq_au'], 298.15, 101325)

# =======================================================
# ТЕРМОХІМІЧНІ ПАРАМЕТРИ
# =======================================================
T = results['temperature'][0]
P = results['pressure'][0]

print("\n" + "="*80)
print(f"{'ТЕРМОХІМІЧНІ ПАРАМЕТРИ':^80}")
print("="*80)
print(f"Температура (T):             {T:10.2f} {results['temperature'][1]}")
print(f"Тиск (P):                    {P:10.2f} {results['pressure'][1]}")
print(f"Ротаційні константи [{results['rot_const'][1]}]: "
      f"{results['rot_const'][0][0]:10.5f}  {results['rot_const'][0][1]:10.5f}  {results['rot_const'][0][2]:10.5f}")
print(f"Симетрійне число:            {results['sym_number'][0]:>10d}")
print(f"Нульова коливальна енергія (ZPE): {results['ZPE'][0]:10.5f} Eh"
      f" = {results['ZPE'][0] * nist.HARTREE2J * nist.AVOGADRO:12.3f} Дж/моль")

# ===============================================================
#  ТЕРМОДИНАМІЧНА ТА ЕНЕРГЕТИЧНА ІНФОРМАЦІЯ
# ===============================================================

# -------------------------------------------------------
# ТЕРМОДИНАМІЧНА ТАБЛИЦЯ
# -------------------------------------------------------
keys = ('tot', 'elec', 'trans', 'rot', 'vib')
header = ["Функція", "Одиниці"] + [x.upper() for x in keys]

def convert(f, keys, unit):
    """Переведення значень термодинамічних функцій у потрібні одиниці"""
    conv = nist.HARTREE2J * nist.AVOGADRO if 'Eh' in unit else 1
    return [results.get(f + '_' + key, (0,))[0] * conv for key in keys]

def write_table_row(title, f):
    """Створює один рядок термодинамічної таблиці"""
    tot, unit = results[f + '_tot']
    values = convert(f, keys, unit)
    # Заміна одиниць для більш зрозумілого відображення
    unit = unit.replace('Eh', 'J/mol·K')
    return [title, unit] + [f"{v:10.3f}" for v in values]

# Формуємо таблицю для S, Cv, Cp
table = [
    write_table_row("Ентропія S", "S"),
    write_table_row("Cv", "Cv"),
    write_table_row("Cp", "Cp"),
]

print("\n" + "="*100)
print(f"{f'ТЕРМОДИНАМІЧНІ ФУНКЦІЇ (T = {T:.2f} K, P = {P/101325:.2f} atm)':^100}")
print("="*100)
print(f"{header[0]:<15s} {header[1]:<15s} " + " ".join(f"{h:>10s}" for h in header[2:]))
print("-"*100)
for row in table:
    print(f"{row[0]:<15s} {row[1]:<15s} " + " ".join(row[2:]))
print("="*100)

# -------------------------------------------------------
# ЕНЕРГЕТИЧНА ТАБЛИЦЯ (у kJ/mol)
# -------------------------------------------------------
Ha2kJ = nist.HARTREE2J * nist.AVOGADRO / 1000  # 1 Eh → kJ/mol

def convert_kJ(f, keys):
    """Повертає список енергій у kJ/mol для заданої функції."""
    return [results.get(f + '_' + key, (0,))[0] * Ha2kJ for key in keys]

def write_energy_row_kJ(title, f):
    values = convert_kJ(f, keys)
    return [title, "kJ/mol"] + [f"{v:12.3f}" for v in values]

# --- ZPE ---
ZPE_kJ = results['ZPE'][0] * Ha2kJ
ZPE_row = ["ZPE", "kJ/mol", f"{ZPE_kJ:12.3f}"]

energy_table_kJ = [
    ZPE_row,
    write_energy_row_kJ("E", "E"),
    write_energy_row_kJ("H", "H"),
    write_energy_row_kJ("G", "G"),
]

print("\n" + "="*100)
print(f"{f'ЕНЕРГЕТИЧНІ ВЕЛИЧИНИ (T = {T:.2f} K, P = {P/101325:.2f} atm)':^100}")
print("="*100)
print(f"{header[0]:<15s} {header[1]:<15s} " + " ".join(f"{h:>12s}" for h in header[2:]))
print("-"*100)
for row in energy_table_kJ:
    print(f"{row[0]:<15s} {row[1]:<15s} " + " ".join(row[2:]))
print("="*100)

