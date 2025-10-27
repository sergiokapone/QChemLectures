from pyscf import gto, scf
from pyscf.scf import addons

# ========== ТРИПЛЕТ (1s2s ³S) ==========
mol_triplet = gto.M(
    atom = 'He 0 0 0',
    basis = 'cc-pvtz',
    spin = 2
)
mf_triplet = scf.UHF(mol_triplet).run(verbose=0)
e_triplet = mf_triplet.e_tot

# ========== СИНГЛЕТ ЗБУДЖЕНИЙ (1s2s ¹S) З MOM ==========
mol_singlet = gto.M(
    atom = 'He 0 0 0',
    basis = 'cc-pvtz',
    spin = 0
)

# Спочатку UHF для триплету як initial guess
mf_init = scf.UHF(mol_singlet).run(verbose=0)

# Модифікуємо occupation для alpha і beta
mo_occ = mf_init.mo_occ.copy()
mo_occ[0][0] = 1  # alpha: 1s
mo_occ[0][1] = 1  # alpha: 2s
mo_occ[1][0] = 0  # beta: порожньо
mo_occ[1][1] = 0  # beta: порожньо

# Новий розрахунок з MOM
mf_singlet = scf.UHF(mol_singlet)
mf_singlet = addons.mom_occ(mf_singlet, mf_init.mo_coeff, mo_occ).run(verbose=0)
e_singlet = mf_singlet.e_tot

# ========== ПОРІВНЯННЯ ==========
exp_triplet = -2.145974
exp_singlet = -2.123843

print(f'\n{"="*65}')
print(f'{"Стан":<15} {"HF (Ha)":>12} {"Експ. (Ha)":>12} {"Δ (Ha)":>10}')
print(f'{"-"*65}')
print(f'{"Триплет ³S":<15} {e_triplet:>12.6f} {exp_triplet:>12.6f} {abs(e_triplet - exp_triplet):>10.6f}')
print(f'{"Синглет ¹S":<15} {e_singlet:>12.6f} {exp_singlet:>12.6f} {abs(e_singlet - exp_singlet):>10.6f}')
print(f'{"="*65}')

