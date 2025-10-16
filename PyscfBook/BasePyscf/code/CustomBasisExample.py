from pyscf import gto

# Визначення базису для H (тільки s-функції)
mol = gto.M(
    atom='H 0 0 0',
    basis={
        'H': gto.basis.parse('''
            H    S
                 13.00773     0.019685
                  1.962079    0.137977
                  0.444529    0.478148
            H    S
                  0.1219492   1.000000
        ''')
    }
)

print(f'Власний базис: {mol.nao_nr()} функцій')
