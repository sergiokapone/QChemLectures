from pyscf import gto, scf
import numpy as np

# Дані про атоми другого періоду
atoms_data = [
    ("Li", 1, "²S", "[He] 2s¹"),
    ("Be", 0, "¹S", "[He] 2s²"),
    ("B", 1, "²P", "[He] 2s² 2p¹"),
    ("C", 2, "³P", "[He] 2s² 2p²"),
    ("N", 3, "⁴S", "[He] 2s² 2p³"),
    ("O", 2, "³P", "[He] 2s² 2p⁴"),
    ("F", 1, "²P", "[He] 2s² 2p⁵"),
    ("Ne", 0, "¹S", "[He] 2s² 2p⁶"),
]

basis = "cc-pvdz"
print(f"Розрахунок атомів 2-го періоду (базис: {basis})")
print("=" * 70)
print(f"{'Атом':4s} {'Спін':4s} {'Терм':4s} {'E(HF), Ha':15s} {'E(HF), eV':12s}")
print("-" * 70)

energies = {}

for symbol, spin, term, config in atoms_data:
    mol = gto.M(
        atom=f"{symbol} 0 0 0", basis=basis, spin=spin, symmetry=True, verbose=0
    )

    # Вибір методу SCF
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)

    mf.verbose = 0
    mf.conv_tol = 1e-10
    energy = mf.kernel()
    energies[symbol] = energy

    e_ev = energy * 27.211386
    print(f"{symbol:4s} {spin:4d} {term:4s} {energy:15.8f} {e_ev:12.2f}")

    # Додаткова інформація для відкритих оболонок
    if spin > 0:
        s2 = mf.spin_square()
        expected_s2 = spin * (spin + 2) / 4
        print(f"     <S²> = {s2[0]:.4f} (очікується {expected_s2:.4f})")

print("=" * 70)

# Збереження результатів
np.savez("second_period_hf.npz", **energies)
