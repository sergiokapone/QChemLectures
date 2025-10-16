from pyscf import gto

# Атом H: S=1/2, 2S=1, дублет (M=2)
h = gto.M(atom='H 0 0 0', basis='cc-pvdz', spin=1)

# Атом He: S=0, 2S=0, синглет (M=1)
he = gto.M(atom='He 0 0 0', basis='cc-pvdz', spin=0)

# Атом O: основний стан ³P, S=1, 2S=2, триплет (M=3)
o = gto.M(atom='O 0 0 0', basis='cc-pvdz', spin=2)

# Атом N: основний стан ⁴S, S=3/2, 2S=3, квартет (M=4)
n = gto.M(atom='N 0 0 0', basis='cc-pvdz', spin=3)

print(f'H:  {h.nelec}, 2S={h.spin}, M={h.spin+1}')
print(f'He: {he.nelec}, 2S={he.spin}, M={he.spin+1}')
print(f'O:  {o.nelec}, 2S={o.spin}, M={o.spin+1}')
print(f'N:  {n.nelec}, 2S={n.spin}, M={n.spin+1}')
