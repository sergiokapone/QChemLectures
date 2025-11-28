import numpy as np
from pyscf import gto, scf
from collections import defaultdict

DIMER_DISTANCES_HF_STO3G = {
    'HH+': 2.0,    # Точне значення (один електрон)
    'HH': 1.346,
    'LiLi': 5.5,
    'LiH': 3.015,
    'BB': 3.1,
    'CC': 2.4,
    'NN': 2.0,
    'OO': 2.4,
    'FF': 2.7,
    'NaNa': 6.0,
}

# ---------- Налаштування ----------
elementA = "H"
elementB = 'H'
R_bohr = DIMER_DISTANCES_HF_STO3G.get(f"{elementA}{elementB}+")
basis = "sto-3g"
threshold = 0.3
# -----------------------------------


# Молекула
mol = gto.M(
    atom=f"{elementA} 0 0 -{R_bohr/2}; {elementB} 0 0 {R_bohr/2}",
    basis=basis,
    unit='Bohr',
    verbose=0,
    charge=1,
    spin=1
)
mf = scf.RHF(mol).run()

ao_labels = mol.ao_labels()
C = mf.mo_coeff
E_mo = mf.mo_energy
occ = mf.mo_occ

print("=" * 80)
print(f"MOLECULAR ORBITALS FOR {elementA}-{elementB} (basis: {basis}, R = {R_bohr:.2f} Bohr)")
print("=" * 80)


# Парсимо AO labels
ao_data = []
for i, lab in enumerate(ao_labels):
    parts = lab.split()
    atom_idx = int(parts[0])
    orb_type = parts[2]
    ao_data.append({
        'idx': i,
        'atom_idx': atom_idx,
        'orb': orb_type
    })

def classify_mo_by_overlap(c, ao_data, threshold=0.001):
    """Класифікує МО за знаками коефіцієнтів (перекриванням)"""

    # Групуємо по типу орбіталі
    orb_groups = defaultdict(lambda: [0, 0])

    for i, coeff in enumerate(c):
        if abs(coeff) > threshold:
            orb = ao_data[i]['orb']
            atom_idx = ao_data[i]['atom_idx']
            orb_groups[orb][atom_idx] = coeff

    # Аналізуємо знаки
    bonding_count = 0
    antibonding_count = 0
    terms = []

    for orb in sorted(orb_groups.keys()):
        c_a, c_b = orb_groups[orb]

        # Пропускаємо малі коефіцієнти
        if abs(c_a) < threshold and abs(c_b) < threshold:

            continue

        # Перевіряємо знаки
        same_sign = (c_a * c_b) > 0  # Добуток додатний → однакові знаки
        similar_magnitude = abs(abs(c_a) - abs(c_b)) < threshold

        if similar_magnitude and (abs(c_a) > threshold and abs(c_b) > threshold):
            if same_sign:  # Bonding: + + або - -
                avg = (c_a + c_b) / 2
                terms.append(f"{avg:.3f}({orb}_a + {orb}_b)")
                bonding_count += 1
            else:  # Antibonding: + - або - +
                avg = (c_a - c_b) / 2
                terms.append(f"{avg:.3f}({orb}_a - {orb}_b)")
                antibonding_count += 1
        else:  # Асиметричний внесок
            if abs(c_a) > threshold:
                terms.append(f"{c_a:.3f}·{orb}_a")
            if abs(c_b) > threshold:
                terms.append(f"{c_b:+.3f}·{orb}_b")

    # Визначаємо домінантний характер
    if antibonding_count > bonding_count:
        mo_type = 'antibonding'
    elif bonding_count > antibonding_count:
        mo_type = 'bonding'
    elif bonding_count == 0 and antibonding_count == 0:
        mo_type = 'non-bonding'
    else:
        mo_type = 'mixed'

    expr = " + ".join(terms).replace("+ -", "- ")

    return expr, mo_type

# Використання:
n_mo = C.shape[1]
for mo_idx in range(n_mo):
    c = C[:, mo_idx]
    occ_str = "occ" if occ[mo_idx] > 0 else "virt"

    expr, mo_type = classify_mo_by_overlap(c, ao_data, threshold)

    print(f"\n({occ_str}, {mo_type}) φ{mo_idx+1} = {expr}")
    print(f"     E = {E_mo[mo_idx]:.6f} Ha")

