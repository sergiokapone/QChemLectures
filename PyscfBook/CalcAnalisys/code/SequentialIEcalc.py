from pyscf import gto, scf


def multiple_ionization_energies(symbol, charges, spins, basis="cc-pvtz"):
    """
    Розрахунок послідовних енергій іонізації

    Parameters:
    -----------
    symbol : str
        Символ атома
    charges : list
        Список зарядів [0, 1, 2, ...]
    spins : list
        Відповідні спіни
    """

    print(f"\nПослідовні енергії іонізації {symbol} (базис: {basis})")
    print("=" * 70)

    energies = []

    # Розрахунок енергій для кожного заряду
    for charge, spin in zip(charges, spins):
        mol = gto.M(
            atom=f"{symbol} 0 0 0", basis=basis, charge=charge, spin=spin, verbose=0
        )

        if spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)

        mf.verbose = 0
        mf.conv_tol = 1e-10
        e = mf.kernel()

        energies.append(e)

        ion_state = f"{symbol}{'+' * charge}" if charge > 0 else symbol
        print(f"E({ion_state}): {e:.10f} Ha")

    # Енергії іонізації
    print("\nЕнергії іонізації (eV):")
    for i in range(len(energies) - 1):
        ie = (energies[i + 1] - energies[i]) * 27.211386
        print(f"IE{i + 1} ({symbol}{'+' * i} → {symbol}{'+' * (i + 1)}): {ie:.4f} eV")

    return energies


# Приклади

# Карбон: C, C+, C2+, C3+
print("\n--- Карбон ---")
e_c = multiple_ionization_energies("C", charges=[0, 1, 2, 3], spins=[2, 1, 0, 1])

# Неон: Ne, Ne+, Ne2+
print("\n--- Неон ---")
e_ne = multiple_ionization_energies("Ne", charges=[0, 1, 2], spins=[0, 1, 0])

# Літій: Li, Li+, Li2+, Li3+
print("\n--- Літій ---")
e_li = multiple_ionization_energies("Li", charges=[0, 1, 2, 3], spins=[1, 0, 1, 0])
