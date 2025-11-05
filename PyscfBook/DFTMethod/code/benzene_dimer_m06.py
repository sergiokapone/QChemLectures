# ============================================================
# benzene_dimer_m06.py
# Порівняння M06-функціоналів на паралельно-зміщеній бензеновій димері
# ============================================================
from pyscf import gto, dft
import numpy as np

# Бензен: приблизна геометрія в одній площині
b1 = """
C     0.000   1.396   0.000
C    -1.209   0.698   0.000
C    -1.209  -0.698   0.000
C     0.000  -1.396   0.000
C     1.209  -0.698   0.000
C     1.209   0.698   0.000
H     0.000   2.479   0.000
H    -2.147   1.240   0.000
H    -2.147  -1.240   0.000
H     0.000  -2.479   0.000
H     2.147  -1.240   0.000
H     2.147   1.240   0.000
"""

# Другий бензен: зміщення по x та підняття по z
dx = 1.6   # зміщення (Å)
dz = 3.4   # відстань між площинами (Å)
def translate(fragment, dx, dz):
    lines = []
    for L in fragment.splitlines():
        if not L.strip():
            continue
        parts = L.split()
        x = float(parts[1]) + dx
        y = float(parts[2])
        z = float(parts[3]) + dz
        lines.append(f"{parts[0]} {x:.6f} {y:.6f} {z:.6f}")
    return "\n".join(lines)

b2 = translate(b1, dx=dx, dz=dz)
geom = b1 + "\n" + b2

# Побудова молекули
mol = gto.M(atom=geom, basis='def2-tzvp', unit='Angstrom', verbose=0)

# Список функціоналів для порівняння
xc_list = ['m06-l', 'm06', 'm06-2x']

results = {}
for xc in xc_list:
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.verbose = 0
    e_dimer = mf.kernel()
    results[xc] = {'dimer': e_dimer}

print("Повна енергія димера (Ha):")
for xc, vals in results.items():
    print(f"{xc:8s}  {vals['dimer']:.8f}")
    
