# ============================================================
# h2o_electric_field_gradient.py
# Градієнт електричного поля на ядрах
# ============================================================

from pyscf import gto, scf
import numpy as np

mol = gto.M(
    atom="""
    O  0.0000  0.0000  0.1173
    H  0.0000  0.7572 -0.4692
    H  0.0000 -0.7572 -0.4692
    """,
    basis="6-311g**",  # Потрібен непоганий базис
    unit="angstrom",
)

print("Градієнт електричного поля (EFG) для H2O")
print("=" * 60)

# SCF розрахунок
mf = scf.RHF(mol)
mf.verbose = 0
mf.kernel()

print("\nОбчислення EFG на ядрах...")

# Функція для обчислення EFG
from pyscf.prop import efg

efg_obj = efg.rhf.EFG(mf)
efg_tensors = efg_obj.kernel()

print("\nТензор EFG на атомі O (au):")
print("(V_ij = ∂²V/∂r_i∂r_j)")

efg_O = efg_tensors[0]
print("\n       x          y          z")
for i, label in enumerate(['x', 'y', 'z']):
    print(f"{label}  ", end="")
    for j in range(3):
        print(f"{efg_O[i,j]:>10.4f} ", end="")
    print()

# Діагоналізація для головних компонент
eigvals, eigvecs = np.linalg.eigh(efg_O)

# Сортуємо за |V_ii|
idx = np.argsort(np.abs(eigvals))
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

V_xx, V_yy, V_zz = eigvals

print("\nГоловні компоненти:")
print(f"  V_xx = {V_xx:.6f} au")
print(f"  V_yy = {V_yy:.6f} au")
print(f"  V_zz = {V_zz:.6f} au")
print(f"\nПеревірка (має бути ≈0): V_xx + V_yy + V_zz = {V_xx+V_yy+V_zz:.2e}")

# Параметр асиметрії
eta = (V_xx - V_yy) / V_zz if V_zz != 0 else 0
print(f"\nПараметр асиметрії: η = {eta:.4f}")

# Квадрупольна константа зв'язку (для ЯКР)
# e²qQ/h = eQ × V_zz (потрібен квадрупольний момент ядра)
Q_O17 = -0.02558  # barn для ¹⁷O
e2qQ_MHz = Q_O17 * V_zz * 234.9647  # Конверсія в МГц

print(f"\nКвадрупольна константа для ¹⁷O:")
print(f"  e²qQ/h = {e2qQ_MHz:.3f} МГц")

print("\nЗастосування:")
print("- Ядерний квадрупольний резонанс (ЯКР)")
print("- Інтерпретація Мессбауер спектрів")
print("- Розрахунок констант спін-спінової взаємодії в ЯМР")


