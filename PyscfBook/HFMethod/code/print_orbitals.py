import numpy as np
from pyscf import gto, scf

def print_mo(mol, mf, nmo=5, nao_print=None):
    """
    Друкує молекулярні орбіталі

    Параметри:
    -----------
    mol : gto.Mole object
    mf : SCF object
    nmo : int, кількість МО для друку
    nao_print : int, кількість АТ для друку (None = всі)
    """
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ

    nmo = min(nmo, len(mo_energy))
    if nao_print is None:
        nao_print = mo_coeff.shape[0]

    print("\n" + "="*80)
    print("MOLECULAR ORBITALS".center(80))
    print("="*80)

    # Друкуємо блоками по 5 орбіталей
    for start_mo in range(0, nmo, 5):
        end_mo = min(start_mo + 5, nmo)
        n_cols = end_mo - start_mo

        # Title: MO numbers
        header = "                 "
        for i in range(start_mo, end_mo):
            header += f"{i:>12}"
        print("\n" + header)

        # Енергії орбіталей
        energy_line = "                 "
        for i in range(start_mo, end_mo):
            energy_line += f"{mo_energy[i]:>12.5f}"
        print(energy_line)

        # Заселеності
        occ_line = "                 "
        for i in range(start_mo, end_mo):
            occ_line += f"{mo_occ[i]:>12.5f}"
        print(occ_line)

        # Роздільник
        sep_line = "                 "
        for i in range(n_cols):
            sep_line += "  ----------"
        print(sep_line)

        # Коефіцієнти АО
        ao_labels = mol.ao_labels()
        for iao in range(min(nao_print, mo_coeff.shape[0])):
            # Форматуємо мітку АО
            label = ao_labels[iao]
            # Прибираємо зайві пробіли та форматуємо
            label_parts = label.split()
            if len(label_parts) >= 3:
                atom_label = f"{label_parts[1]}{label_parts[0]}"
                ao_label = label_parts[2]
                formatted_label = f"{atom_label:>4} {ao_label:<10}"
            else:
                formatted_label = f"{label:<15}"

            line = formatted_label + "  "
            for imo in range(start_mo, end_mo):
                line += f"{mo_coeff[iao, imo]:>12.6f}"
            print(line)

    print("\n" + "="*80 + "\n")

mol = gto.Mole(
    atom = '''
    Li 0 0 0
    ''',
    basis = 'sto-3g',
    spin=1
)

mol.build()
mf = scf.RHF(mol)
mf.kernel(verbode=0)


print_mo(mol, mf, nmo=5)
