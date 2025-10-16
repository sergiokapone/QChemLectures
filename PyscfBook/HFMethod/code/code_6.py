from pyscf import gto, scf


def he_excited_state(config="1s2s", spin=0, basis="cc-pvdz"):
    """
    Розрахунок збуджених станів He (¹S, ³S) з використанням MOM
    """
    mol = gto.M(atom="He 0 0 0", basis=basis, spin=spin)

    # RHF для синглету, UHF для триплету
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)

    mf.verbose = 0
    mf.kernel()  # звичайний HF, щоб ініціалізуватись

    # MOM для стабілізації збудженого стану
    mom_solver = scf.addons.mom_occ(mf, mf.make_rdm1(), mf.mo_occ)
    mom_solver.verbose = 0
    mom_solver.max_cycle = 100
    e_exc = mom_solver.kernel()

    return e_exc


# Основний стан
mol_gs = gto.M(atom="He 0 0 0", basis="cc-pvdz", spin=0)
mf_gs = scf.RHF(mol_gs)
mf_gs.verbose = 0
e_gs = mf_gs.kernel()

print("Стани атома Гелію (cc-pVDZ):")
print("-" * 55)
print(f"1s² (¹S, основний): {e_gs:.6f} Ha  = {e_gs * 27.2114:.2f} eV")

# Збуджені стани
for spin, label in [(0, "1s2s (¹S)"), (2, "1s2s (³S)")]:
    e_exc = he_excited_state(spin=spin)
    print(f"{label:15s}: {e_exc: .6f} Ha  ({(e_exc - e_gs)*27.2114: .3f} eV вище основного)")

# Примітка: для збуджених станів
# краще використовувати методи типу TD-DFT або CASSCF
