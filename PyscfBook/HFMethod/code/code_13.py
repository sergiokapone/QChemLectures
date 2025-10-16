from pyscf import gto, scf
import numpy as np
import matplotlib.pyplot as plt


def spherical_averaged_density(symbol, spin, basis="cc-pvdz"):
    """Сферично усереднена електронна густина"""

    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)

    mf.verbose = 0
    mf.kernel()

    dm = mf.make_rdm1()

    # Радіальні точки
    r_points = np.linspace(0.01, 8, 300)

    # Кути для інтегрування (метод Монте-Карло)
    n_angles = 100
    theta = np.random.uniform(0, np.pi, n_angles)
    phi = np.random.uniform(0, 2 * np.pi, n_angles)

    density_sph = []

    for r in r_points:
        rho_avg = 0

        for t, p in zip(theta, phi):
            x = r * np.sin(t) * np.cos(p)
            y = r * np.sin(t) * np.sin(p)
            z = r * np.cos(t)

            coords = np.array([[x, y, z]])
            ao_value = mol.eval_gto("GTOval_sph", coords)

            if spin == 0:
                rho_point = np.einsum("pi,ij,pj->p", ao_value, dm, ao_value)[0]
            else:
                rho_a = np.einsum("pi,ij,pj->p", ao_value, dm[0], ao_value)[0]
                rho_b = np.einsum("pi,ij,pj->p", ao_value, dm[1], ao_value)[0]
                rho_point = rho_a + rho_b

            rho_avg += rho_point

        rho_avg /= n_angles
        density_sph.append(rho_avg)

    # Радіальна функція розподілу 4πr²ρ(r)
    radial_dist = 4 * np.pi * r_points**2 * np.array(density_sph)

    # Графіки
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Густина ρ(r)
    ax1.plot(r_points, density_sph, linewidth=2)
    ax1.set_xlabel("r (Bohr)", fontsize=12)
    ax1.set_ylabel("ρ(r) (e/Bohr³)", fontsize=12)
    ax1.set_title(f"Електронна густина {symbol}", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Радіальна функція розподілу
    ax2.plot(r_points, radial_dist, linewidth=2, color="red")
    ax2.set_xlabel("r (Bohr)", fontsize=12)
    ax2.set_ylabel("4πr²ρ(r)", fontsize=12)
    ax2.set_title(f"Радіальна функція розподілу {symbol}", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{symbol}_spherical_density.pdf")
    plt.show()

    # Перевірка нормування
    dr = r_points[1] - r_points[0]
    total_electrons = np.sum(radial_dist) * dr
    print(f"\nІнтеграл 4πr²ρ(r)dr = {total_electrons:.2f}")
    print(f"Очікується: {mol.nelectron} електронів")

    return r_points, density_sph, radial_dist


# Приклади для різних атомів
r, rho, rdf = spherical_averaged_density("He", spin=0)
r, rho, rdf = spherical_averaged_density("C", spin=2)
r, rho, rdf = spherical_averaged_density("Ne", spin=0)
