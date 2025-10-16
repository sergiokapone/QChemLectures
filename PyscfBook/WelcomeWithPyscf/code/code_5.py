from pyscf import lib
# Конвертація енергії
energy_ha = -2.85516
energy_ev = energy_ha * lib.param.HARTREE2EV
energy_kcal = energy_ha * lib.param.HARTREE2KCAL
print(f'{energy_ha:.6f} Ha = {energy_ev:.4f} eV')
print(f'{energy_ha:.6f} Ha = {energy_kcal:.2f} kcal/mol')
# Конвертація відстані
dist_bohr = 2.0
dist_angstrom = dist_bohr * lib.param.BOHR
print(f'{dist_bohr:.2f} Bohr = {dist_angstrom:.4f} Å')