# ============================================================
# analyze_transitions.py
# Детальний аналіз характеру електронних переходів
# ============================================================

from pyscf import gto, dft, tddft
import numpy as np

mol = gto.M(
    atom="""
    C  0.0000  0.0000  0.0000
    O  0.0000  0.0000  1.2050
    H  0.0000  0.9428 -0.5876
    H  0.0000 -0.9428 -0.5876
    """,
    basis="aug-cc-pvdz",
    unit="angstrom",
)

print("Детальний аналіз переходів H2CO")
print("=" * 60)

# DFT + TDDFT
mf = dft.RKS(mol)
mf.xc = "cam-b3lyp"
mf.verbose = 0
mf.kernel()

td = tddft.TDDFT(mf)
td.nstates = 5
td.verbose = 0
td.kernel()

# Аналізуємо кожен збуджений стан
for i in range(min(3, td.nstates)):  # Перші 3 стани
    energy_ev = td.e[i] * 27.2114
    wavelength = 1240 / energy_ev
    f = td.oscillator_strength()[i]

    print(f"\n{'='*60}")
    print(f"Збуджений стан {i+1}: {energy_ev:.3f} eV ({wavelength:.1f} nm)")
    print(f"Сила осцилятора f = {f:.4f}")
    print(f"{'='*60}")

    # Отримуємо амплітуди переходів X → Y
    # td.xy[i] містить (X, Y) амплітуди
    x_amp = td.xy[i][0]  # Амплітуди збудження

    # Індекси орбіталей
    nocc = mol.nelectron // 2  # Кількість зайнятих орбіталей

    print("\nОсновні внески (|амплітуда| > 0.1):")
    print(f"{'Перехід':<20} {'Амплітуда':<12} {'Внесок (%)'}")
    print("-" * 50)

    # Знаходимо значущі внески
    nvir = len(x_amp[0])  # Кількість віртуальних

    contributions = []
    for occ_i in range(nocc):
        for vir_i in range(nvir):
            amp = x_amp[occ_i, vir_i]
            contrib = amp**2 * 100  # У відсотках

            if abs(amp) > 0.1:
                homo_label = occ_i - nocc  # -1 для HOMO, -2 для HOMO-1, ...
                lumo_label = vir_i  # 0 для LUMO, 1 для LUMO+1, ...

                if homo_label == -1:
                    orb_from = "HOMO"
                elif homo_label < -1:
                    orb_from = f"HOMO{homo_label+1}"
                else:
                    orb_from = f"Occ{occ_i}"

                if lumo_label == 0:
                    orb_to = "LUMO"
                else:
                    orb_to = f"LUMO+{lumo_label}"

                transition = f"{orb_from} → {orb_to}"
                contributions.append((transition, amp, contrib))

    # Сортуємо за внеском
    contributions.sort(key=lambda x: x[2], reverse=True)

    for trans, amp, contrib in contributions[:5]:  # Топ-5
        print(f"{trans:<20} {amp:>10.4f}  {contrib:>10.1f}")

    # Інтерпретація
    if contributions:
        main_trans = contributions[0][0]
        print(f"\nДомінуючий перехід: {main_trans}")

        if "HOMO" in main_trans and "LUMO" in main_trans:
            print("Тип: одноелектронне збудження")
            if i == 0:
                print("Характер: n→π* (неподілена пара O → π* C=O)")
            elif i == 1:
                print("Характер: π→π* (зв'язуюча → антизв'язуюча)")

print("\n" + "="*60)
print("Примітка: Внески показують ймовірність кожного переходу")
print("Сума квадратів амплітуд дорівнює 1")

