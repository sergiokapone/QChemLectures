# =========================================================
# FullCI_basic.py — повний CI для молекули H2
# =========================================================

from pyscf import gto, scf, fci

# Вибір базису
basis="cc-pv5z"

# --- 1. Створюємо молекулу
mol = gto.M(
    atom = "He 0 0 0",  # відстань 0.74 Å
    basis=basis,
    verbose = 0
)

# --- 2. Розрахунок Хартрі–Фока
mf = scf.RHF(mol)
E_HF = mf.kernel()

# --- 3. Повний CI
cisolver = fci.FCI(mf)
E_FCI, _ = cisolver.kernel()

# --- 4. Вивід результатів
E_exp = -2.9037243770  # Експериментальне значення (basis set limit)
abs_E_exp = abs(E_exp)
dev_HF = abs(E_HF - E_exp) / abs_E_exp * 100
dev_FCI = abs(E_FCI - E_exp) / abs_E_exp * 100

# Вивід таблиці
print("=" * 60)
header = f"{{:<{12}}} {{:>{14}}} {{:>{12}}}"
print(header.format("Метод", "Енергія (Ha)", "Відхилення (%)"))
print("=" * 60)
row = f"{{:<{12}}} {{:>{14}.{8}f}} {{:>{12}.{4}f}}"
print(row.format("HF", E_HF, dev_HF))
print(row.format("Full CI", E_FCI, dev_FCI))
print(row.format("Експеримент", E_exp, 0.0000))
print("=" * 60)
