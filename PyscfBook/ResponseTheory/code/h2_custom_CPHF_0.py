import numpy as np
from pyscf import gto, scf
from pyscf.prop.polarizability.rhf import Polarizability

# === H2O ===
mol = gto.M(
    atom='''
    O 0.000000 0.000000 0.000000
    H 0.000000 -0.757000 0.587000
    H 0.000000 0.757000 0.587000
    ''',
    basis='6-31g',
    verbose=0
)

mf = scf.RHF(mol).run(conv_tol=1e-12)
print(f"Енергія: {mf.e_tot:.10f} Hartree")

mo_coeff = mf.mo_coeff
mo_energy = mf.mo_energy
mo_occ = mf.mo_occ
nocc = int(np.sum(mo_occ > 0))
nvir = len(mo_energy) - nocc
ov = nocc * nvir

print("="*60)
print("CPHF ПОЛЯРИЗОВАНІСТЬ — ФІНАЛЬНА ВЕРСІЯ")
print("="*60)
print(f"nocc = {nocc}, nvir = {nvir}, ov = {ov}\n")

# === Диполь ===
with mol.with_common_orig((0,0,0)):
    dip_ao = mol.intor_symmetric('int1e_r', comp=3)
dip_mo = np.einsum('pi,xpq,qj->xij', mo_coeff, dip_ao, mo_coeff)

# === ERI ===
C_occ = mo_coeff[:, :nocc]
C_vir = mo_coeff[:, nocc:]
eri_ao = mol.intor('int2e')

print("ERI → MO...")
eri_iajb = np.einsum('pi,pqrs->iqrs', C_occ, eri_ao)
eri_iajb = np.einsum('qa,iqrs->iars', C_vir, eri_iajb)
eri_iajb = np.einsum('rj,iars->iajs', C_occ, eri_iajb)
eri_iajb = np.einsum('sb,iajs->iajb', C_vir, eri_iajb)  # (i,a,j,b)

eri_ijab = np.einsum('pi,pqrs->iqrs', C_occ, eri_ao)
eri_ijab = np.einsum('qj,iqrs->ijrs', C_occ, eri_ijab)
eri_ijab = np.einsum('ra,ijrs->ijas', C_vir, eri_ijab)
eri_ijab = np.einsum('sb,ijas->ijab', C_vir, eri_ijab)  # (i,j,a,b)

eri_ibja = eri_iajb.transpose(0, 3, 2, 1)

# === H ===
print("H матриця...")
i_occ = np.arange(nocc).repeat(nvir)
a_vir = np.tile(np.arange(nvir), nocc)
H = np.diag(mo_energy[nocc + a_vir] - mo_energy[i_occ])
H += 4.0 * eri_iajb.reshape(ov, ov)
H -= eri_ijab.transpose(0, 2, 1, 3).reshape(ov, ov)
H -= eri_ibja.transpose(0, 3, 2, 1).reshape(ov, ov)
print(f"H: {H.shape}")

# === CPHF: H U = g, g = +μ_ov ===
g_ov = -dip_mo[:, :nocc, nocc:]   # +μ, не -μ!
U = np.zeros((3, nocc, nvir))

print("\nCPHF...")
for x in range(3):
    U[x] = np.linalg.solve(H, g_ov[x].ravel()).reshape(nocc, nvir)
    print(f"  {['x','y','z'][x]}: ||U|| = {np.linalg.norm(U[x]):.6f}")

# === α_xy = –2 Σ (μ^x U^y + μ^y U^x) ===
mu_ov = dip_mo[:, :nocc, nocc:]  # (3, nocc, nvir)
alpha = np.zeros((3, 3))
for x in range(3):
    for y in range(3):
        alpha[x, y] = -2.0 * (np.sum(mu_ov[x] * U[y]) + np.sum(mu_ov[y] * U[x]))
alpha = 0.5 * (alpha + alpha.T)

# === Вивід ===
print("\n" + "="*60)
print("НАШ РЕЗУЛЬТАТ")
print(alpha.round(8))
print(f"⟨α⟩ = {np.trace(alpha)/3:.6f} a.u.")

# === PySCF ===
print("\n" + "="*60)
pol = Polarizability(mf)
alpha_ref = pol.polarizability()
print("PySCF:")
print(alpha_ref.round(8))
print("Різниця:")
print((alpha - alpha_ref).round(8))
print(f"max|Δ| = {np.max(np.abs(alpha - alpha_ref)):.2e}")

