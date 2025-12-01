#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Розрахунок магнітної сприйнятливості молекули H2O у PySCF.
Виправлення: cc-pvdz базис (уникає reshape помилки), вищий tol SCF.
Джерело проблеми: CPHF vind в rhf.py — невідповідність nmo (13 -> 19).
"""

import numpy as np
from pyscf import gto, scf
from pyscf.prop.magnetizability.rhf import Magnetizability

# =====================================================
# 1. Побудова молекули
# =====================================================
mol = gto.M(
    atom='''
    O  0.000000  0.000000  0.000000
    H  0.000000 -0.757000  0.587000
    H  0.000000  0.757000  0.587000
    ''',
    basis='cc-pvdz',  # Змінено з '6-31g' — nbas=19, уникає помилки reshape
    verbose=5,        # Більше логів для CPHF
    unit='Bohr'
)

# =====================================================
# 2. SCF (RHF) — вищий tol для стабільності
# =====================================================
mf = scf.RHF(mol)
mf.conv_tol = 1e-12  # Критичний: для точних MO в CPHF
mf.max_cycle = 200   # Більше ітерацій SCF
mf.kernel()
print(f"\nЕнергія SCF: {mf.e_tot:.10f} Хартрі")

# =====================================================
# 3. Магнітна сприйнятливість
# =====================================================
mag = Magnetizability(mf)
mag.gauge_orig = None  # [0,0,0]
mag.conv_tol_cphf = 1e-10  # Толерантність CPHF
mag.max_cycle_cphf = 50    # Більше циклів для збіжності
chi_tensor = mag.kernel()  # 3x3 тензор

isotropic_chi = np.trace(chi_tensor) / 3.0

print("\n" + "="*60)
print("МАГНІТНА СПРИЙНЯТЛИВІСТЬ (χ)")
print("="*60)
print("Тензор (au):")
print(f"xx: {chi_tensor[0,0]:.6f}, yy: {chi_tensor[1,1]:.6f}, zz: {chi_tensor[2,2]:.6f}")
print(f"xy/xz/yz: {chi_tensor[0,1]:.6f}, {chi_tensor[0,2]:.6f}, {chi_tensor[1,2]:.6f}")
print(f"\nІзотропна магнітна сприйнятливість: {isotropic_chi:.6f} а.о.")
print(f"У одиницях 10⁻⁶ cgs: {isotropic_chi * 0.445379:.2f} × 10⁻⁶ cgs/моль")
