# =========================================================
# LiH_size_consistency.py — демонстрація size-consistency для LiH дисоціації з експериментом
# =========================================================

from pyscf import gto, scf
from pyscf.fci import FCI
from pyscf.ci import CISD

# Атом Li (UHF для open-shell, spin=1)
mol_li = gto.M(atom="Li 0 0 0", spin=1, basis="sto-3g", verbose=0)
mf_li = scf.UHF(mol_li)
E_hf_li = mf_li.kernel()
fci_li = FCI(mf_li)
E_fci_li = fci_li.kernel()[0]
cisd_li = CISD(mf_li)
E_cisd_li_corr = cisd_li.kernel()[0]

# Атом H (UHF для open-shell, spin=1)
mol_h = gto.M(atom="H 0 0 0", spin=1, basis="sto-3g", verbose=0)
mf_h = scf.UHF(mol_h)
E_hf_h = mf_h.kernel()
# FCI та CISD для H = HF (1 електрон, немає кореляції)

# LiH bound (R=3.0 Bohr ≈ рівноважна, RHF, spin=0)
mol_bound = gto.M(atom="Li 0 0 0; H 0 0 3.0", basis="sto-3g", verbose=0)
mf_bound = scf.RHF(mol_bound)
E_hf_bound = mf_bound.kernel()
fci_bound = FCI(mf_bound)
E_fci_bound = fci_bound.kernel()[0]
cisd_bound = CISD(mf_bound)
E_cisd_bound_corr = cisd_bound.kernel()[0]

# LiH dissociated (R=10, RHF — для демонстрації проблеми)
mol_diss = gto.M(atom="Li 0 0 0; H 0 0 20", basis="sto-3g", verbose=0)
mf_diss = scf.RHF(mol_diss)
E_hf_diss = mf_diss.kernel()
fci_diss = FCI(mf_diss)
E_fci_diss = fci_diss.kernel()[0]
cisd_diss = CISD(mf_diss)
E_cisd_diss_corr = cisd_diss.kernel()[0]

# Li + H (граничний стан)
E_2atoms_hf = E_hf_li + E_hf_h
E_2atoms_fci = E_fci_li + E_hf_h  # H exact
E_2atoms_cisd = (E_hf_li + E_cisd_li_corr) + E_hf_h  # Li correlation

# Експериментальні значення (приблизно для R_eq ≈3.0 Bohr)
E_exp_bound = -8.070  # Експериментальна енергія зв'язаної LiH (Ha)
E_exp_diss = -7.978  # E(Li) + E(H) (Ha)

# Повні енергії CISD
E_cisd_bound = E_hf_bound + E_cisd_bound_corr
E_cisd_diss = E_hf_diss + E_cisd_diss_corr

# Помилки size-consistency
diff_hf = E_hf_diss - E_2atoms_hf
diff_fci = E_fci_diss - E_2atoms_fci
diff_cisd = E_cisd_diss - E_2atoms_cisd
diff_exp = 0  # ≈0

print("="*65)
print("Метод       | LiH bound | LiH diss  | Li + H   | Error (diss - 2atoms)")
print("="*65)
print(f"HF          | {E_hf_bound:8.5f}  | {E_hf_diss:8.5f}  | {E_2atoms_hf:8.5f} | {diff_hf:8.5f}")
print(f"CISD        | {E_cisd_bound:8.5f}  | {E_cisd_diss:8.5f}  | {E_2atoms_cisd:8.5f} | {diff_cisd:8.5f}")
print(f"Full CI     | {E_fci_bound:8.5f}  | {E_fci_diss:8.5f}  | {E_2atoms_fci:8.5f} | {diff_fci:8.5f}")
print(f"Експеримент | {E_exp_bound:8.5f}  | {E_exp_diss:8.5f}  | {E_exp_diss:8.5f} | {diff_exp:8.5f}")
print("="*65)
