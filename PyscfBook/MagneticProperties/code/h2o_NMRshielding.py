#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Розрахунок тензорів хімічного екранування (NMR shielding)
та хімічних зсувів для молекули H2O.

Виправлення: gauge_orig = None (або (0,0,0)), бо 'GIAO' не підтримується як рядок.
GIAO вбудовано для para частини. Використовуйте центр заряду для gauge-independence.
Базис: cc-pvdz (уникає CPHF помилок з 6-31G).
"""

import numpy as np
from pyscf import gto, scf
from pyscf.prop.nmr.rhf import NMR

# =====================================================
# 1. Побудова молекули
# =====================================================
mol = gto.M(
    atom='''
    O  0.000000  0.000000  0.000000
    H  0.000000 -0.757000  0.587000
    H  0.000000  0.757000  0.587000
    ''',
    basis='cc-pvdz',  # Змінено з '6-31g' — краща точність, nbas=19 (без reshape помилок)
    verbose=5,        # Логи для CPHF
    unit='Bohr'
)

# =====================================================
# 2. SCF (RHF) — вищий tol
# =====================================================
mf = scf.RHF(mol)
mf.conv_tol = 1e-12  # Для точних MO в CPHF
mf.max_cycle = 200
mf.kernel()
print(f"\nЕнергія SCF: {mf.e_tot:.10f} Гартрі")

# =====================================================
# 3. NMR: тензори екранування
# =====================================================
nmr = NMR(mf)
nmr.gauge_orig = None  # Common origin [0,0,0]; GIAO вбудовано
# Альтернатива: центр заряду (gauge-independent)
# nmr.gauge_orig = mol.atom_chirc()  # ≈ (0,0,0.39) для H2O
nmr.conv_tol_cphf = 1e-10  # Толерантність CPHF
nmr.max_cycle_cphf = 50    # Більше циклів
sigma_tensors = nmr.kernel()  # Список 3x3 тензорів для кожного атома

# =====================================================
# 4. Ізотропні значення та хімічні зсуви
# =====================================================
atom_symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
sigma_iso = [np.trace(sig) / 3.0 for sig in sigma_tensors]

# Референсні значення (газова фаза, для cc-pVDZ; з літератури, e.g., Helgaker)
ref_shielding = {'H': 31.0, 'O': -72.0}  # ppm (приблизно для H2O)

print("\n" + "="*60)
print("ХІМІЧНИЙ ЗСУВ (NMR)")
print("="*60)

chemical_shifts = []
for i, (sym, sig_iso) in enumerate(zip(atom_symbols, sigma_iso)):
    ref = ref_shielding.get(sym, 0.0)
    delta = ref - sig_iso
    chemical_shifts.append(delta)

    print(f"Атом {i+1} ({sym}):")
    print(f"   σ_iso = {sig_iso:8.3f} ppm")
    print(f"   δ     = {delta:8.3f} ppm (відносно {sym}-референсу)")

# Середнє для H (симетричні атоми)
h_indices = [i for i, s in enumerate(atom_symbols) if s == 'H']
if h_indices:
    avg_H = np.mean([chemical_shifts[i] for i in h_indices])
    print(f"\nСередній хімічний зсув для H: {avg_H:.3f} ppm")
    
