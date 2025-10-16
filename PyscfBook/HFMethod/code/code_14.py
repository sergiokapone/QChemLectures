from pyscf import gto, scf
import numpy as np
import matplotlib.pyplot as plt


def compare_densities(atoms_list, basis="cc-pvdz"):
    """Порівняння електронної густини різних атомів"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(atoms_list)))

    for (symbol, spin), color in zip(atoms_list, colors):
        mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

        if spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)

        mf.verbose = 0
        mf.kernel()

        dm = mf.make_rdm1()

        # Густина вздовж осі z
        r_points = np.linspace(0.01, 6, 200)
        density = []

        for r in r_points:
            coords = np.array([[0, 0, r]])
            ao = mol.eval_gto("GTOval_sph", coords)

            if spin == 0:
                rho = np.einsum("pi,ij,pj->p", ao, dm, ao)[0]
            else:
                rho = (
                    np.einsum("pi,ij,pj->p", ao, dm[0], ao)[0]
                    + np.einsum("pi,ij,pj->p", ao, dm[1], ao)[0]
                )

            density.append(rho)

        # Графіки
        ax1.plot(
            r_points,
            density,
            linewidth=2,
            label=f"{symbol} (Z={mol.atom_charge(0)})",
            color=color,
        )

        # Радіальна функція
        radial = 4 * np.pi * r_points**2 * np.array(density)
        ax2.plot(r_points, radial, linewidth=2, label=f"{symbol}", color=color)

    ax1.set_xlabel("r (Bohr)", fontsize=12)
    ax1.set_ylabel("ρ(r) (e/Bohr³)", fontsize=12)
    ax1.set_title("Електронна густина", fontsize=14)
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("r (Bohr)", fontsize=12)
    ax2.set_ylabel("4πr²ρ(r)", fontsize=12)
    ax2.set_title("Радіальний розподіл", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("atoms_density_comparison.pdf")
    plt.show()


# Порівняння атомів благородних газів
noble_gases = [("He", 0), ("Ne", 0), ("Ar", 0)]
compare_densities(noble_gases, basis="cc-pvtz")

# Порівняння атомів 2-го періоду
second_period = [("Li", 1), ("C", 2), ("N", 3), ("O", 2), ("F", 1)]
compare_densities(second_period, basis="6-31g*")
