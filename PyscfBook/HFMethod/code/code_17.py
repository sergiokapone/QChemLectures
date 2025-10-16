from pyscf import gto, scf

mol = gto.M(atom="Co 0 0 0", basis="def2-svp", spin=3)
mf = scf.UHF(mol).run()
print("Converged?", mf.converged)
