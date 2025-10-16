from pyscf import gto, scf, mcscf
import numpy as np


def casscf_transition_metal(symbol, spin, nelecas, ncas, basis="def2-svp"):
    """
    CASSCF для перехідного металу
    """

    mol = gto.M(
        atom=f"{symbol} 0 0 0",
        basis=basis,
        spin=spin,
        symmetry=False,  # Часто простіше без симетрії
        verbose=0,
    )

    print(f"\nCASCF розрахунок {symbol} (2S={spin})")
    print(f"Активний простір: CASSCF({nelecas},{ncas})")
    print("=" * 70)

    # UHF
    mf = scf.UHF(mol)
    mf.verbose = 0
    mf.conv_tol = 1e-10
    e_hf = mf.kernel()

    print(f"UHF енергія: {e_hf:.10f} Ha")

    s2_hf = mf.spin_square()[0]
    expected_s2 = spin * (spin + 2) / 4
    print(f"<S²> (UHF): {s2_hf:.6f} (очікується {expected_s2:.6f})")

    # CASSCF
    print(f"\nCASCF({nelecas},{ncas}) розрахунок...")
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.verbose = 4
    mc.conv_tol = 1e-8
    mc.max_cycle_macro = 100

    # Для важких випадків
    mc.fcisolver.max_cycle = 100
    mc.fcisolver.conv_tol = 1e-8

    try:
        e_casscf = mc.kernel()[0]

        print(f"\nCASCF енергія: {e_casscf:.10f} Ha")
        print(f"Статична кореляція: {(e_casscf - e_hf) * 1000:.6f} mHa")

        # Природні орбіталі
        natocc, natorb = mc.cas_natorb()

        print("\nЗаселеності природних орбіталей (активний простір):")
        for i, occ in enumerate(natocc):
            if occ > 0.01:  # тільки значні заселеності
                print(f"  Орбіталь {i + 1}: {occ:.4f}")

        # Оцінка багатоконфігураційності
        # Strongly occupied (> 1.98) та weakly occupied (< 0.02)
        n_strong = np.sum(natocc > 1.98)
        n_weak = np.sum(natocc < 0.02)
        n_fractional = ncas - n_strong - n_weak

        print(f"\nАналіз заселеностей:")
        print(f"  Сильно заповнені (>1.98): {n_strong}")
        print(f"  Дробові (0.02-1.98): {n_fractional}")
        print(f"  Порожні (<0.02): {n_weak}")

        if n_fractional > ncas / 2:
            print("  → Сильна багатоконфігураційність!")

        return e_hf, e_casscf, natocc

    except Exception as e:
        print(f"\nПомилка: {str(e)}")
        print("Спробуйте інший активний простір або початкове наближення")
        return e_hf, None, None


# Приклади

# Cr: [Ar] 3d⁵ 4s¹
# CASSCF(6,6): 6 електронів у 6 орбіталях (5×3d + 1×4s)
e_hf_cr, e_cas_cr, occ_cr = casscf_transition_metal(
    "Cr", spin=6, nelecas=6, ncas=6, basis="def2-svp"
)

# Fe: [Ar] 3d⁶ 4s²
# CASSCF(8,5): 8 електронів у 5 d-орбіталях
e_hf_fe, e_cas_fe, occ_fe = casscf_transition_metal(
    "Fe", spin=4, nelecas=8, ncas=5, basis="def2-svp"
)
