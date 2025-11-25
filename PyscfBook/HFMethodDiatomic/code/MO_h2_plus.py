import numpy as np
from pyscf import gto, scf
from collections import defaultdict

# ---------- Налаштування ----------
element = "H"
R_bohr =  2
basis = "sto-3g"
threshold = 0.001  # мінімальний коефіцієнт для виводу
# -----------------------------------

# Побудова молекули та SCF
mol = gto.M(
    atom=f"{element} 0 0 -{R_bohr/2}; {element} 0 0 {R_bohr/2}",
    basis=basis,
    unit='Bohr',
    verbose=0,
    charge=1,
    spin=1
)
mf = scf.RHF(mol).run()

# Зчитування даних
ao_labels = mol.ao_labels()
C = mf.mo_coeff
E = mf.mo_energy
occ = mf.mo_occ

print("=" * 80)
print(f"MOLECULAR ORBITALS FOR {element}₂ (basis: {basis}, R = {R_bohr:.2f} Bohr)")
print("=" * 80)

# Парсимо AO labels
ao_data = []
for i, lab in enumerate(ao_labels):
    parts = lab.split()
    atom_idx = int(parts[0])  # 0 або 1
    atom_name = parts[1]      # Li, Be, etc.
    orb_type = parts[2]       # 1s, 2s, 2px, etc.
    ao_data.append({
        'idx': i,
        'atom_idx': atom_idx,
        'atom_name': atom_name,
        'orb': orb_type
    })

def format_mo(c, ao_data, threshold=0.001):
    """Формат з групуванням атомів"""

    # Групуємо по типу орбіталі
    orb_groups = defaultdict(lambda: [0, 0])

    for i, coeff in enumerate(c):
        if abs(coeff) > threshold:
            orb = ao_data[i]['orb']
            atom_idx = ao_data[i]['atom_idx']
            orb_groups[orb][atom_idx] = coeff

    # Формуємо компактний вираз
    terms = []
    bond_types = []  # зберігаємо типи зв'язків
    for orb in sorted(orb_groups.keys()):
        c_a, c_b = orb_groups[orb]

        if abs(c_a - c_b) < 0.01:  # Майже однакові (bonding)
            avg = (c_a + c_b) / 2
            if abs(avg) > threshold:
                terms.append(f"{avg:.3f}({orb}_(a) + {orb}_(b))")
                bond_types.append('bonding')
        elif abs(c_a + c_b) < 0.01:  # Майже протилежні (antibonding)
            avg = (c_a - c_b) / 2
            if abs(avg) > threshold:
                terms.append(f"{avg:.3f}({orb}_(a) - {orb}_(b))")
                bond_types.append('antibonding')
        else:  # Змішані
            if abs(c_a) > threshold:
                terms.append(f"{c_a:.3f}·{orb}_(a)")
            if abs(c_b) > threshold:
                terms.append(f"{c_b:+.3f}·{orb}_(b)")
                bond_types.append('mixed')

    expr = " + ".join(terms).replace("+ -", "- ")

    # Визначаємо домінантний тип
    if 'antibonding' in bond_types:
        mo_type = 'antibonding'
    elif 'bonding' in bond_types:
        mo_type = 'bonding'
    else:
        mo_type = 'mixed'

    return expr, mo_type

# Виводимо кожну МО
n_mo = C.shape[1]
for mo_idx in range(n_mo):
    c = C[:, mo_idx]
    c_filtered = np.where(np.abs(c) > threshold, c, 0)
    occ_str = "occ" if occ[mo_idx] > 0 else "virt"

    expr, mo_type = format_mo(c_filtered, ao_data, threshold)
    print(f"\n({occ_str}, {mo_type}) φ{mo_idx+1} = {expr}")
    print(f"     (ΔE = {E[mo_idx] - (-1.0):.6f} Ha)")

print("\n" + "=" * 80)

