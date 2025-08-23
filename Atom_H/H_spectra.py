from pyscf import gto, scf

import matplotlib.pyplot as plt
import numpy as np


mol = gto.M(atom='H 0 0 0', basis='cc-pVQZ', unit='Bohr', spin=1)
mf = scf.UHF(mol)
mf.kernel()

# Энергии орбиталей
print(mf.mo_energy)

# Энергии МО в eV
mo_e = mf.mo_energy * 27.211386245988

print(mo_e)

# # Все переходы
# transitions = []
# for i in range(len(mo_e)):
#     for j in range(i+1, len(mo_e)):
#         transitions.append(mo_e[j] - mo_e[i])

# plt.vlines(transitions, 0, 1)
# plt.xlabel("Энергия перехода, eV")
# plt.ylabel("Интенсивность (условная)")
# plt.title("Пример спектра H-атома (HF)")
# plt.show()
