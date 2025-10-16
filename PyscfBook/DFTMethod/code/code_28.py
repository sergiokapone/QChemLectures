from pyscf import gto, scf, dft
import time


def timing_comparison(symbol, spin, basis="def2-tzvp"):
    """
    Порівняння часу виконання HF vs DFT
    """

    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)

    methods = {"HF": None, "LDA": "lda", "PBE": "pbe", "B3LYP": "b3lyp", "PBE0": "pbe0"}

    print(f"\nЧас виконання для {symbol} (базис: {basis})")
    print("=" * 60)
    print(f"{'Метод':10s} {'Час, с':10s} {'Відносно HF':15s}")
    print("-" * 60)

    times = {}

    for method, xc in methods.items():
        if method == "HF":
            mf = scf.UHF(mol) if spin > 0 else scf.RHF(mol)
        else:
            mf = dft.UKS(mol) if spin > 0 else dft.RKS(mol)
            mf.xc = xc

        mf.verbose = 0

        # Вимірювання часу
        start = time.time()
        mf.kernel()
        elapsed = time.time() - start

        times[method] = elapsed

        if method == "HF":
            t_ref = elapsed
            rel = 1.0
        else:
            rel = elapsed / t_ref

        print(f"{method:10s} {elapsed:10.4f} {rel:15.3f}x")

    print("=" * 60)


# Тестування
timing_comparison("C", spin=2, basis="cc-pvtz")
timing_comparison("Fe", spin=4, basis="def2-svp")
timing_comparison("Kr", spin=0, basis="def2-tzvp")
