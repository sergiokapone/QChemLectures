from pyscf import gto, scf

mol = gto.M(atom='V 0 0 0', basis='cc-pvdz', spin=3)
mf = scf.UHF(mol)

# Callback для моніторингу збіжності
def monitor(envs):
    cycle = envs['cycle']
    e_tot = envs['e_tot']
    norm_gorb = envs['norm_gorb']
    print(f'Cycle {cycle:2d}: E={e_tot:.8f}, |grad|={norm_gorb:.2e}')

mf.callback = monitor
energy = mf.kernel()

# Після завершення:
print(f'\nФінальна енергія: {energy:.8f} Ha')
print(f'Число ітерацій: {mf.iterations}')
print(f'Норма градієнта: {mf.scf_summary["norm_gorb"]:.2e}')
