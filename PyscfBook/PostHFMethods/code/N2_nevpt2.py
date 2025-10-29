# ============================================================
# NEVPT2 поверх CASSCF: приклад для молекули N2
# ============================================================
from pyscf import gto, scf, mcscf, mrpt

# 1. Молекула
mol = gto.M(
    atom='N 0 0 0; N 0 0 1.1',
    basis='cc-pvdz',
    symmetry=True,
    verbose=0
)

# 2. Hartree-Fock
mf = scf.RHF(mol)
mf.verbose = 0
e_hf = mf.kernel()

# 3. CASSCF
mc = mcscf.CASSCF(mf, ncas=6, nelecas=6)
mc.verbose = 0
e_casscf = mc.kernel()[0]

# 4. NEVPT2
pt = mrpt.NEVPT(mc)
pt.verbose = 0
e_nevpt2_corr = pt.kernel()
e_nevpt2_tot = e_casscf + e_nevpt2_corr

# Аналіз кореляції
e_static = e_casscf - e_hf           # Статична (CASSCF)
e_dynamic = e_nevpt2_corr            # Динамічна (NEVPT2)
e_total_corr = e_static + e_dynamic  # Повна кореляція

# Відсотки
static_pct = abs(e_static / e_total_corr * 100)
dynamic_pct = abs(e_dynamic / e_total_corr * 100)

# Вивід
print("\n" + "="*70)
print("  NEVPT2 для молекули N₂")
print("="*70)
print(f"  Базис: cc-pVDZ  |  Активний простір: CAS(6,6)")
print(f"  Відстань N-N: 1.1 Å")
print("-"*70)
print("  Енергії:")
print(f"    E(HF)      = {e_hf:16.8f} Ha")
print(f"    E(CASSCF)  = {e_casscf:16.8f} Ha")
print(f"    E(NEVPT2)  = {e_nevpt2_tot:16.8f} Ha")
print("="*70)
print("  Кореляційна енергія:")
print(f"    Статична  (CASSCF)     = {e_static:12.6f} Ha  ({static_pct:5.1f}%)")
print(f"    Динамічна (NEVPT2)     = {e_dynamic:12.6f} Ha  ({dynamic_pct:5.1f}%)")
print(f"    Повна кореляція        = {e_total_corr:12.6f} Ha  (100.0%)")
print("="*70)
print(f"  Відновлено {static_pct:.1f}% статичної + {dynamic_pct:.1f}% динамічної кореляції")
print("="*70 + "\n")

