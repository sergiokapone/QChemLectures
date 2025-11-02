"""
tddft_h2o_analysis.py
=====================

Розрахунок збуджених станів молекули H₂O у наближенні TDHF (Random Phase Approximation, RPA)
методами бібліотеки PySCF.


Опис:
------
Цей скрипт виконує такі кроки:
1. Будує молекулу води з базисним набором 6-31G.
2. Розв’язує рівняння Хартрі–Фока для отримання основного стану.
3. Виконує TDHF (RPA) обчислення для п’яти низькоенергетичних збуджених станів.
4. Виводить:
   • енергію HF;
   • енергії збуджених станів (у Hartree та eV);
   • осциляторні сили та переходні дипольні моменти;
   • найсильніші одноелектронні переходи (i → a) у кожному збудженому стані;
   • частку зворотного компонента Y/X (оцінка RPA-релаксації).

Фізичне значення:
-----------------
TDHF дає перше наближення до електронних збуджень — тобто УФ і VUV переходів.
Для H₂O очікуються три основні смуги:
    ~7.4 eV  — 1b₁ → 4a₁ (основний УФ-перехід),
    ~9–10 eV — n → σ* збудження,
    >12 eV   — VUV-область (високоенергетичні переходи).

Вихідні дані дозволяють співставити частоти з експериментальними спектрами та
оцінити силу переходів через осциляторні сили.

"""
import numpy as np
from pyscf import gto, scf
from pyscf.data import nist

# 1. Задаємо молекулу
mol = gto.Mole()
mol.atom = '''
O  0.000000  0.000000  0.000000
H  0.000000  0.757160  0.586260
H  0.000000 -0.757160  0.586260
'''
mol.basis = '6-31G'
mol.verbose = 0
mol.build()

# 2. Розв'язуємо рівняння Хартрі–Фока
mf = scf.RHF(mol).run()
print(f"Енергія HF = {mf.e_tot:.6f} Hartree\n")

# 3. TDHF (Random Phase Approximation)
td = mf.TDHF()
td.nstates = 5
td.kernel()

# 4. Енергії збуджених станів
print("=== Збуджені стани TDHF ===")
for i, e in enumerate(td.e):
    print(f"Стан {i+1}:  ω = {e:.6f} Hartree  = {e*nist.HARTREE2EV:.3f} eV")

# 4а. Осциляторні сили та дипольні моменти
osc_strengths = td.oscillator_strength()
dip_moments = td.transition_dipole()

print("\n=== Осциляторні сили та дипольні моменти ===")
for i, (fval, mu_vec, e) in enumerate(zip(osc_strengths, dip_moments, td.e), start=1):
    mu_norm = np.linalg.norm(mu_vec)
    print(f"Стан {i}: ω = {e*27.2114:6.3f} eV  "
          f"|μ| = {mu_norm:6.3f} a.u.  f = {fval:7.4f}")

# 5. Аналіз збуджених станів
print("\n=== Аналіз збуджених станів ===")

mo_occ = mf.mo_occ
mo_energy = mf.mo_energy
nocc = np.count_nonzero(mo_occ > 0)
nvir = len(mo_occ) - nocc

for i, (omega, (X, Y)) in enumerate(zip(td.e, td.xy), start=1):
    print(f"\nСтан {i}: ω = {omega:.6f} Hartree = {omega*nist.HARTREE2EV:.3f} eV")

    # 1) Нормалізація (|X|² − |Y|²)
    norm = np.linalg.norm(X)**2 - np.linalg.norm(Y)**2
    print(f"  Нормалізація (|X|²−|Y|²) = {norm:.4f}")

    # 2) Визначення найсильніших переходів
    Xmat = X.reshape(nocc, nvir)
    idx = np.unravel_index(np.argsort(abs(Xmat.ravel()))[::-1][:3], Xmat.shape)
    print("  Топ-3 переходи:")
    for k in range(3):
        i_occ, a_vir = idx[0][k], idx[1][k]
        e_occ = mo_energy[i_occ]
        e_vir = mo_energy[nocc + a_vir]
        delta_e = (e_vir - e_occ) * 27.2114
        print(f"    {k+1}) i={i_occ:2d} → a={a_vir:2d} "
              f"Δε={delta_e:6.2f} eV  |X|={abs(Xmat[i_occ,a_vir]):.4f}")

    # 3) Частка зворотного компонента
    y_ratio = np.linalg.norm(Y) / np.linalg.norm(X)
    print(f"  Частка зворотного компонента |Y|/|X| = {y_ratio:.3f}")

