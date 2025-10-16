from pyscf import gto, scf

atoms_open_shell = [
    ("H", 1),
    ("Li", 1),
    ("B", 1),
    ("C", 2),
    ("N", 3),
    ("O", 2),
    ("F", 1),
]

basis = "6-31g*"

print(f"Порівняння UHF vs ROHF (базис: {basis})")
print("=" * 120)
print(f"{'Атом':4s} {'Спін':4s} {'E(UHF), Ha':15s} {'E(ROHF), Ha':15s} {'ΔE, mHa':10s} {'⟨S²⟩UHF':10s} {'ΔS²':8s} {'Оцінка':40s}")
print("-" * 120)

for symbol, spin in atoms_open_shell:
    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

    # UHF
    mf_uhf = scf.UHF(mol)
    mf_uhf.conv_tol = 1e-10
    e_uhf = mf_uhf.kernel()

    # ROHF
    mf_rohf = scf.ROHF(mol)
    mf_rohf.conv_tol = 1e-10
    e_rohf = mf_rohf.kernel()

    # Спінові характеристики
    s2_uhf, s_uhf = mf_uhf.spin_square()
    s2_rohf, s_rohf = mf_rohf.spin_square()

    # Теоретичне значення S(S+1)
    S = spin / 2
    S2_expected = S * (S + 1)

    delta_e = (e_uhf - e_rohf) * 1000  # mHa
    delta_s2 = s2_uhf - S2_expected

    # Класифікація за критеріями
    if delta_s2 < 0.05:
        quality = "незначне забруднення, прийнятно"
    elif delta_s2 < 0.10:
        quality = "помірне, бажано перевірити"
    else:
        quality = "суттєве забруднення, бажано ROHF або проєкційні методи"

    print(f"{symbol:4s} {spin:4d} {e_uhf:15.8f} {e_rohf:15.8f} {delta_e:10.4f} {s2_uhf:10.4f} {delta_s2:8.4f} {quality:40s}")

print("=" * 120)
print("\nПримітка:")
print("UHF часто дає нижчу енергію, але може мати спінове забруднення (ΔS²).")
print("ROHF зберігає спінову симетрію, але іноді трохи вищу енергію.")

