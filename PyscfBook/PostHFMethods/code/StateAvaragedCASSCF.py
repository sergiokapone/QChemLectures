"""
StateAveragedCASSCF.py
State-averaged CASSCF для різних станів Карбону
"""

from pyscf import gto, scf, mcscf

mol = gto.M(
    atom="C 0 0 0",
    basis="cc-pvtz",
    spin=2,
    symmetry=True,
    verbose=0,
)

# HF
mf = scf.UHF(mol)
mf.verbose = 0
e_hf = mf.kernel()

# SA-CASSCF
mc = mcscf.CASSCF(mf, 4, 4)
mc.verbose = 0  # Вимкнути весь verbose!
mc.state_average_([0.5, 0.5])
mc.conv_tol = 1e-9
e_sa = mc.kernel()[0]

# Гарний вивід
print("\n" + "="*60)
print("  State-Averaged CASSCF для атома Карбону")
print("="*60)
print(f"  Базис: cc-pVTZ  |  Активний простір: CAS(4,4)")
print("  Стани: ³P (основний) + ¹D (збуджений)")
print("-"*60)
print(f"  E(UHF)           = {e_hf:14.8f} Ha")
print(f"  E(SA-CASSCF)     = {e_sa:14.8f} Ha")
print(f"  Кореляція        = {e_sa - e_hf:14.8f} Ha")
print("="*60)
print("  Енергії окремих станів:")
print("-"*60)

for i, e_state in enumerate(mc.e_states):
    state_name = "³P" if i == 0 else "¹D"
    e_ev = e_state * 27.211386
    print(f"    Стан {i+1} ({state_name:2s}):  {e_state:14.8f} Ha  ({e_ev:10.4f} eV)")

if len(mc.e_states) > 1:
    delta_e_ha = mc.e_states[1] - mc.e_states[0]
    delta_e_ev = delta_e_ha * 27.211386
    delta_e_cm = delta_e_ev * 8065.54  # eV → cm⁻¹
    print("-"*60)
    print(f"  Енергія збудження ³P → ¹D:")
    print(f"    ΔE = {delta_e_ha:10.6f} Ha  =  {delta_e_ev:8.4f} eV  =  {delta_e_cm:10.1f} cm⁻¹")

print("="*60 + "\n")


def sa_casscf_carbon():
    """
    State-averaged CASSCF для різних станів Карбону
    """

    mol = gto.M(
        atom="C 0 0 0",
        basis="cc-pvtz",
        spin=2,  # Для триплету
        symmetry=True,
        verbose=0,
    )

    print("\nState-Averaged CASSCF для C")
    print("Стани: ³P (основний) та ¹D (збуджений)")
    print("=" * 70)

    # HF
    mf = scf.UHF(mol)
    mf.verbose = 0
    e_hf = mf.kernel()

    print(f"UHF енергія: {e_hf:.10f} Ha")

    # SA-CASSCF(4,4): усереднення по декількох станах
    mc = mcscf.CASSCF(mf, 4, 4)
    mc.verbose = 4

    # State-averaging для 2 станів з рівними вагами
    mc.state_average_([0.5, 0.5])

    # Або для різних симетрій:
    # mc = mc.state_average_mix_([
    #     mcscf.state_average_(mc, [0.5, 0.5]),  # стани симетрії 1
    # ])

    mc.conv_tol = 1e-9
    e_sa = mc.kernel()[0]

    print(f"\nSA-CASSCF енергія (усереднена): {e_sa:.10f} Ha")

    # Енергії окремих станів
    print("\nЕнергії окремих станів:")
    for i, e_state in enumerate(mc.e_states):
        print(f"  Стан {i + 1}: {e_state:.10f} Ha ({e_state * 27.211386:.4f} eV)")

    # Різниця енергій
    if len(mc.e_states) > 1:
        delta_e = (mc.e_states[1] - mc.e_states[0]) * 27.211386
        print(f"\nРізниця енергій збудження: {delta_e:.4f} eV")


sa_casscf_carbon()
