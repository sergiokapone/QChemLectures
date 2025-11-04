# ============================================================
# h2o_raman_full.py
# Розрахунок Раман-активностей H2O + порівняння з експериментом
# ============================================================

from pyscf import gto, scf
from pyscf.hessian import rhf as rhf_h
from pyscf.prop.polarizability import rhf as pol_rhf
import numpy as np

# ------------------------------------------------------------
# Геометрія H2O
# ------------------------------------------------------------
mol = gto.M(
    atom="""
    O  0.0000  0.0000  0.1173
    H  0.0000  0.7572 -0.4692
    H  0.0000 -0.7572 -0.4692
    """,
    basis="6-31g",
    unit="angstrom",
)
print("Розрахунок Раман-спектру H2O (PySCF, RHF/6-31G)")
print("=" * 70)

# ------------------------------------------------------------
# SCF
# ------------------------------------------------------------
mf = scf.RHF(mol).run()

# ------------------------------------------------------------
# Гессіан та нормальні моди
# ------------------------------------------------------------
hess = rhf_h.Hessian(mf).kernel()
mass = mol.atom_mass_list()
natm = mol.natm

# ------------------------------------------------------------
# Перетворення Гессіана в масо-зважену матрицю
# ------------------------------------------------------------
natm = mol.natm
mass = mol.atom_mass_list()

# "Розплющений" 3N×3N Гессіан
hess_flat = hess.reshape(3*natm, 3*natm)

# Маси (кожен атом по 3 координати)
mvec = np.repeat(np.sqrt(mass), 3)

# Масо-зважена форма
m_hess = hess_flat / np.outer(mvec, mvec)

# Власні значення → частоти
freqs2, modes = np.linalg.eigh(m_hess)
freqs = np.sign(freqs2) * np.sqrt(np.abs(freqs2)) * 5140.48  # см⁻¹


# Відкидаємо 6 нульових мод (поступальні + обертальні)
vib_freqs = freqs[6:]
vib_modes = modes[:, 6:]
vib_modes = vib_modes.reshape(natm, 3, -1)

# ------------------------------------------------------------
# Чисельна похідна поляризовності
# ------------------------------------------------------------
dr = 0.01  # малі відхилення (Å)
polar = pol_rhf.Polarizability(mf)

raman_activity = []
for k in range(vib_modes.shape[2]):
    disp = dr * vib_modes[:, :, k]
    coords0 = mol.atom_coords()

    mol_disp = mol.copy()
    mol_disp.set_geom_(coords0 + disp, unit="angstrom")
    mf_plus = scf.RHF(mol_disp).run()
    alpha_plus = pol_rhf.Polarizability(mf_plus).polarizability()

    mol_disp.set_geom_(coords0 - disp, unit="angstrom")
    mf_minus = scf.RHF(mol_disp).run()
    alpha_minus = pol_rhf.Polarizability(mf_minus).polarizability()

    dalpha = (alpha_plus - alpha_minus) / (2 * dr)
    alpha_mean = np.trace(dalpha) / 3
    anisotropy = np.sqrt(
        0.5 * ((dalpha[0,0]-dalpha[1,1])**2 +
               (dalpha[1,1]-dalpha[2,2])**2 +
               (dalpha[2,2]-dalpha[0,0])**2 +
               6*(dalpha[0,1]**2 + dalpha[1,2]**2 + dalpha[2,0]**2))
    )
    I = 45 * alpha_mean**2 + 7 * anisotropy**2
    raman_activity.append(I)

# ------------------------------------------------------------
# Порівняння з експериментом
# ------------------------------------------------------------
exp_data = [
    ("ν₁ Симетричне розтягування OH", 3657, "сильна"),
    ("ν₂ Деформація HOH", 1595, "середня"),
    ("ν₃ Асиметричне розтягування OH", 3756, "слабка"),
]

print("\nТаблиця 1. Раман-активності H₂O (RHF/6-31G) vs експеримент\n")
print(f"{'Мода':35s} {'Теор. частота (см⁻¹)':>22s} {'Інтенсивність (відн. од.)':>30s} {'Експ. частота (см⁻¹)':>22s} {'Експ. активність':>20s}")
print("-" * 125)

for (label, exp_freq, exp_intensity), freq, I in zip(exp_data, vib_freqs, raman_activity):
    print(f"{label:35s} {freq:22.1f} {I:30.4f} {exp_freq:22d} {exp_intensity:>20s}")

print("\nПримітка: теоретичні частоти без масштабування (≈10–15% завищені).")

