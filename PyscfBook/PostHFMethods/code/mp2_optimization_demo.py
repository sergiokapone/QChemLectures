# ============================================================
# File: mp2_optimization_demo.py
# Оптимізація MP2-розрахунків у PySCF:
#   - базовий MP2
#   - заморожування ядерних орбіталей (frozen core)
#   - використання RI-MP2 (density fitting)
#   - локалізація орбіталей (Pipek–Mezey)
#   - порівняння з експериментом
# ============================================================

from pyscf import gto, scf, mp, lo

# --- 1. Створення молекули ---
mol = gto.M(
    atom = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
    basis = "cc-pVTZ",
    verbose = 0
)

# --- 2. SCF розрахунок (RHF) ---
mf = scf.RHF(mol).run(verbose=0)

# --- 3. Стандартний MP2 (усі орбіталі активні) ---
mp2_full = mp.MP2(mf)
mp2_full.verbose = 0
mp2_full.kernel()
print(f"MP2 (усі орбіталі):     E = {mp2_full.e_tot:.8f} Ha")

# --- 4. MP2 із замороженим ядром ---
mp2_frozen = mp.MP2(mf).set(frozen=2)  # заморожуємо 1s-орбіталі Оксигену
mp2_frozen.verbose = 0
mp2_frozen.kernel()
print(f"MP2 (frozen core):      E = {mp2_frozen.e_tot:.8f} Ha")

# --- 5. RI-MP2 (Resolution of Identity) ---
mf_df = mf.density_fit()  # перехід до скороченого інтегрального представлення
mp2_df = mp.MP2(mf_df)
mp2_df.verbose = 0
mp2_df.kernel()
print(f"RI-MP2 (density fit):   E = {mp2_df.e_tot:.8f} Ha")

# --- 6. Локалізація орбіталей (Pipek–Mezey) ---
loc_orb = lo.PM(mf.mol, mf.mo_coeff).kernel()
print(f"Pipek–Mezey локалізація виконана: {loc_orb.shape[1]} орбіталей")

# --- 7. Порівняння з експериментом ---
E_exp = -76.438  # експериментальна повна енергія води при 0 K (Ha)
print("\n=== Порівняння енергій ===")
print(f"HF       = {mf.e_tot:.8f} Ha")
print(f"MP2(full)= {mp2_full.e_tot:.8f} Ha")
print(f"MP2(froz)= {mp2_frozen.e_tot:.8f} Ha")
print(f"RI-MP2   = {mp2_df.e_tot:.8f} Ha")
print(f"Exp.     = {E_exp:.8f} Ha (експеримент)")

print("\n=== Відхилення від експерименту ===")
print(f"HF        : {mf.e_tot - E_exp:+.6f} Ha")
print(f"MP2(full) : {mp2_full.e_tot - E_exp:+.6f} Ha")
print(f"MP2(froz) : {mp2_frozen.e_tot - E_exp:+.6f} Ha")
print(f"RI-MP2    : {mp2_df.e_tot - E_exp:+.6f} Ha")

