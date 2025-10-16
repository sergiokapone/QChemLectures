# Атом H (1 неспарений електрон): S=1/2, 2S=1
mol = gto.M(atom='H 0 0 0', spin=1)
# Атом He (всі спарені): S=0, 2S=0
mol = gto.M(atom='He 0 0 0', spin=0)
# Атом O у триплетному стані: S=1, 2S=2
mol = gto.M(atom='O 0 0 0', spin=2)