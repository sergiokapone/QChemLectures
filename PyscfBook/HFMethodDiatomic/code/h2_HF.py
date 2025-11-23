# h2_HL_analytical.py — Аналітична крива Гайтлера-Лондона (правильні формули)
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.special import expi


def S_overlap(R, zeta=1.0):
    """Інтеграл перекриття S = <a|b>"""
    return (1 + zeta*R + (zeta*R)**2/3) * np.exp(-zeta*R)

def S_moverlap(R, zeta=1.0):
    """Інтеграл перекриття S = <a|b>"""
    return (1 - zeta*R + (zeta*R)**2/3) * np.exp(zeta*R)

def E_HL(R, zeta=1.0):
    """
    Повна енергія методу Гайтлера-Лондона

    E = [H_aa + H_bb + H_ab + H_ba] / [1 + S²]

    де H_ij = <ij|H|ij> включає всі одно- та двоелектронні доданки
    """
    S = S_overlap(R, zeta)
    Sm = S_moverlap(R, zeta)

    # Енергія ізольованого атома H
    E_atom = -zeta**2 / 2
    Q = 1/ R * np.exp(-2 * R) * (1 + 5/8 * R - 3/4 * R ** 2 - 1/6 * R ** 3)
    M = np.exp(R) * (1 - R + 1 / 3 * R ** 2)
    A = (
        S ** 2  / R - np.exp(-2*R)*(11/8 + 103/20*R + 49/15*R**2 + 11/15*R**3) +
        6/ (5*R) * (S**2 *(0.57722+np.log(R)) + Sm * expi(-4*R) -
        2 * S * Sm * expi(-2*R) )
    )


    return 2*E_atom + (Q + A) / (1 + S ** 2)

if __name__ == "__main__":
    print("="*70)
    print("АНАЛІТИЧНА КРИВА ГАЙТЛЕРА-ЛОНДОНА ДЛЯ H₂")
    print("="*70)

    zeta = 1.0  # Оригінальна експонента (1927)

    # Розрахунок кривої
    Rs = np.linspace(0.25, 10.0, 200)
    Es = [E_HL(R, zeta) for R in Rs]

    # Енергія двох ізольованих атомів
    E_two_atoms = 2 * (-zeta**2 / 2)  # = -1.0 Ha для zeta=1.0

    # Збереження
    with open("h2_hl_analytical.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["R(Bohr)", "E_HL(Ha)"])
        for r, e in zip(Rs, Es):
            w.writerow([r, e])

    # Пошук мінімуму
    idx = np.argmin(Es)
    R_min = Rs[idx]
    E_min = Es[idx]
    D_e = E_two_atoms - E_min

    print(f"\nПараметри розрахунку:")
    print(f"  ζ = {zeta}")
    print(f"  E(H) = {-zeta**2/2:.6f} Ha")
    print(f"  E(2H) = {E_two_atoms:.6f} Ha")

    print(f"\n" + "="*70)
    print("РЕЗУЛЬТАТИ:")
    print("="*70)
    print(f"Рівноважна відстань:")
    print(f"  R_min = {R_min:.4f} Bohr = {R_min*0.529177:.4f} Å")
    print(f"\nМінімальна енергія:")
    print(f"  E_min = {E_min:.6f} Ha")
    print(f"\nЕнергія зв'язку:")
    print(f"  D_e = {D_e:.6f} Ha = {D_e*27.211:.3f} eV")
    print(f"\nАсимптота:")
    print(f"  E(R→∞) = {Es[-1]:.6f} Ha")
    print(f"  Очікувалось: {E_two_atoms:.6f} Ha")

    print(f"\n" + "="*70)
    print("ІСТОРИЧНІ РЕЗУЛЬТАТИ (Heitler & London, 1927):")
    print("="*70)
    print(f"  R_e ≈ 1.64 Bohr (0.87 Å)")
    print(f"  D_e ≈ 3.14 eV (0.115 Ha)")

    print(f"\n" + "="*70)
    print("ЕКСПЕРИМЕНТАЛЬНІ ЗНАЧЕННЯ:")
    print("="*70)
    print(f"  R_e = 1.401 Bohr (0.741 Å)")
    print(f"  D_e = 4.75 eV (0.1745 Ha)")

    # Графік
    plt.figure(figsize=(12, 7))

    plt.plot(Rs, Es, 'b-', linewidth=2.5, label=f'Heitler-London (ζ={zeta})')
    plt.plot(R_min, E_min, 'ro', markersize=10,
             label=f'Мінімум: R={R_min:.3f} Bohr, D_e={D_e*27.211:.2f} eV')

    # Експериментальні значення
    E_exp_min = E_two_atoms - 0.1745
    plt.axhline(y=E_exp_min, color='green', linestyle='--', linewidth=1.5,
                label=f'Експеримент: D_e = 4.75 eV', alpha=0.7)
    plt.axvline(x=1.401, color='green', linestyle='--', linewidth=1.5,
                label='Експеримент: R_e = 1.401 Bohr', alpha=0.7)

    # Історичні значення HL (1927)
    E_hist_min = E_two_atoms - 0.115
    plt.plot(1.64, E_hist_min, 'ms', markersize=10,
             label='Історичний HL (1927): R≈1.64 Bohr, D_e≈3.14 eV')

    # Асимптота
    plt.axhline(y=E_two_atoms, color='red', linestyle=':', linewidth=2,
                label=f'Асимптота: E = {E_two_atoms:.1f} Ha (2 атоми H)', alpha=0.7)

    # Область зв'язку
    plt.fill_between(Rs, Es, E_two_atoms, where=(np.array(Es) < E_two_atoms),
                     alpha=0.1, color='blue', label='Область зв\'язку')

    plt.xlabel('R (Bohr)', fontsize=13)
    plt.ylabel('E (Hartree)', fontsize=13)
    plt.title('Аналітична крива Гайтлера-Лондона для H₂ (1927)',
              fontsize=15, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(0.5, 8)
    plt.ylim(E_min - 0.1, 0.1)
    plt.tight_layout()
    plt.savefig('h2_hl_analytical.png', dpi=300, bbox_inches='tight')

    print(f"\n" + "="*70)
    print("Файли збережено:")
    print("  - h2_hl_analytical.png")
    print("  - h2_hl_analytical.csv")
    print("="*70)
    
