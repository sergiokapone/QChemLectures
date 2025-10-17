import numpy as np

from pyscf import gto, scf

# Створюємо молекулу води
mol = gto.M(
    atom="""
    O  0.0000  0.0000  0.1173
    H  0.0000  0.7572 -0.4692
    H  0.0000 -0.7572 -0.4692
    """,
    basis="6-31g",
    unit="angstrom",
)

print("Молекула H2O")
print("=" * 60)

# Спочатку SCF розрахунок
mf = scf.RHF(mol)
mf.kernel()

print("\nОбчислення гесіану...")
# Аналітичний гесіан (точний і швидкий)
hess = mf.Hessian()
h = hess.kernel()

print(f"Форма гесіану (4D): {h.shape}")
print(f"Гесіан симетричний: {np.allclose(h, np.transpose(h, (1, 0, 3, 2)))}")

# Переформована 9x9 матриця для зручності
h_flat = h.reshape(9, -1)  # Вказуємо одну розмірність явно
print(f"Форма сплощеної матриці: {h_flat.shape}")

# Перетворюємо пласку матрицю у квадратну форму
n = int(np.sqrt(h_flat.size))
h_matrix = h_flat.reshape((n, n))

np.set_printoptions(precision=6, suppress=True, linewidth=120)
print("\nПовна матриця гесіану (Ha/bohr²):")
print(h_flat)

# Виведемо кілька елементів (з урахуванням 4D-структури)
print("\nПриклад елементів гесіану (Ha/bohr²):")
print(f"∂²E/∂x_O ∂x_O = H[0,0,0,0] = {h[0,0,0,0]:.6f}")  # ∂²E/∂x_O ∂x_O
print(f"∂²E/∂x_O ∂y_O = H[0,0,0,1] = {h[0,0,0,1]:.6f}")  # ∂²E/∂x_O ∂y_O
print(f"∂²E/∂y_O ∂y_O = H[0,0,1,1] = {h[0,0,1,1]:.6f}")  # ∂²E/∂y_O ∂y_O

