from pyscf import gto, scf
import matplotlib.pyplot as plt

basis_sets = [
    "sto-3g",
    "3-21g",
    "6-31g",
    "6-311g",
    "cc-pvdz",
    "cc-pvtz",
    "cc-pvqz",
    "cc-pv5z",
]

energies = []
n_basis = []

print("Базис           N_bas    Енергія (Ha)    Похибка (mHa)")
print("-" * 60)

for basis in basis_sets:
    try:
        mol = gto.M(atom="H 0 0 0", basis=basis, spin=1, verbose=0)
        mf = scf.UHF(mol)
        e = mf.kernel()

        n = mol.nao_nr()
        error = (e + 0.5) * 1000  # похибка в міліГартрі

        energies.append(e)
        n_basis.append(n)

        print(f"{basis:15s} {n:3d}    {e:12.8f}    {error:8.4f}")
    except:
        print(f"{basis:15s} --- недоступний")

# Побудова графіка збіжності
plt.figure(figsize=(10, 6))
plt.plot(n_basis, energies, "o-", linewidth=2, markersize=8)
plt.axhline(y=-0.5, color="r", linestyle="--", label="Точне значення")
plt.xlabel("Кількість базисних функцій", fontsize=12)
plt.ylabel("Енергія (Ha)", fontsize=12)
plt.title("Збіжність енергії атома H від базисного набору", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("h_atom_basis_convergence.pdf")
plt.show()
