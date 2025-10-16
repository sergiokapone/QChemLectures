from pyscf import gto, scf
from pyscf.tools import cubegen
import numpy as np
import matplotlib.pyplot as plt


def plot_density_profile(symbol, spin, basis="cc-pvdz"):
    """Радіальний розподіл густини"""

    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)

    mf.verbose = 0
    mf.kernel()

    # Розрахунок густини вздовж осі z
    r_points = np.linspace(0, 5, 200)  # Bohr
    density = []

    dm = mf.make_rdm1()

    for r in r_points:
        coords = np.array([[0, 0, r]])
        rho = mol.eval_gto("GTOval_sph", coords)

        if spin == 0:
            dens = np.einsum("pi,ij,pj->p", rho, dm, rho)[0]
        else:
            dens_a = np.einsum("pi,ij,pj->p", rho, dm[0], rho)[0]
            dens_b = np.einsum("pi,ij,pj->p", rho, dm[1], rho)[0]
            dens = dens_a + dens_b

        density.append(dens)

    # Побудова графіка
    plt.figure(figsize=(10, 6))
    plt.plot(r_points, density, linewidth=2)
    plt.xlabel("Відстань від ядра (Bohr)", fontsize=12)
    plt.ylabel("Електронна густина (e/Bohr³)", fontsize=12)
    plt.title(f"Радіальний розподіл густини {symbol}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{symbol}_density.pdf")
    plt.show()

    return r_points, density


# Приклад
r, rho = plot_density_profile("Ne", spin=0)
