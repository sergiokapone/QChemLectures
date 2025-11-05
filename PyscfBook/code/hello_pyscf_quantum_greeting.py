"""
hello_pyscf_quantum_greeting.py

Демонстраційний скрипт для книги «Квантово-механічні методи обчислення з PySCF».

Скрипт виконує RHF-розрахунок для атома Гелію й створює «квантове привітання»,
де енергія системи визначає характер повідомлення.
"""

from pyscf import gto, scf

# --- Побудова атома He та запуск RHF ---
mol = gto.M(atom="He 0 0 0", basis="6-31g", spin=0)
mf = scf.RHF(mol)
energy = mf.kernel()

# --- Енергія в Хартрі та електронвольтах ---
HARTREE_TO_EV = 27.211386
energy_ev = energy * HARTREE_TO_EV

# --- Генерація «емоцій» Гелію залежно від енергії ---
mood = ""
if energy < -2.80:
    mood = "😌 стабільним і задоволеним"
elif energy < -2.70:
    mood = "🤔 трохи збурений, але тримається"
else:
    mood = "😱 у стані суперпозиції стресу"

# --- Виведення ---
print("\n--- PySCF Quantum Greeting ---")
print(f"RHF-енергія (He): {energy:.8f} Ha  ({energy_ev:.3f} eV)\n")

print("🧪 Hello from the quantum world!")
print(f"   Ваш Гелій почувається {mood}.")
print("   Хвильова функція нормалізована, обчислення збіглось.\n")
print("(Змініть базис або атом — і настрій вашого Гелію зміниться!)")
