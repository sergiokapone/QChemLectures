from pyscf import gto, scf
import time

# -------------------------------------------------------------
# З симетрією
# -------------------------------------------------------------
t0 = time.perf_counter()

mol_sym = gto.M(
    atom='''
    O  0.0000  0.0000  0.0000
    H  0.7570  0.5860  0.0000
    H -0.7570  0.5860  0.0000
    ''',
    basis='cc-pVDZ',
    symmetry=True
)
mf_sym = scf.RHF(mol_sym).run()
t1 = time.perf_counter()

# -------------------------------------------------------------
# Без симетрії
# -------------------------------------------------------------
t2 = time.perf_counter()

mol_nosym = gto.M(
    atom=mol_sym.atom,  # та сама геометрія
    basis='cc-pVDZ',
    symmetry=False
)
mf_nosym = scf.RHF(mol_nosym).run()
t3 = time.perf_counter()

# -------------------------------------------------------------
# Порівняння
# -------------------------------------------------------------
e_sym = mf_sym.e_tot
e_nosym = mf_nosym.e_tot
time_sym = t1 - t0
time_nosym = t3 - t2

print(f'З симетрією:  {e_sym:.8f} Ha  (час: {time_sym:.3f} c)')
print(f'Без симетрії: {e_nosym:.8f} Ha  (час: {time_nosym:.3f} c)')
print(f'Різниця енергій: {abs(e_sym - e_nosym):.2e} Ha')
print(f'Прискорення: {time_nosym / time_sym:.2f}×')
print(f'Виявлена група симетрії: {mol_sym.topgroup}')

