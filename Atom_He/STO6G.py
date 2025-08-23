import numpy as np
import matplotlib.pyplot as plt

# Визначення діапазону r
r = np.linspace(0, 5, 500)

# Нормувальний коефіцієнт для STO-1s
zeta = 1.7
sto_1s = (zeta ** 3 / np.pi) ** 0.5 *  np.exp(-zeta * r)

# 1         0.6598456824E+02       0.9163596281E-02
# 2         0.1209819836E+02       0.4936149294E-01
# 3         0.3384639924E+01       0.1685383049E+00
# 4         0.1162715163E+01       0.3705627997E+00
# 5         0.4515163224E+00       0.4164915298E+00
# 6         0.1859593559E+00       0.1303340841E+00

params = [
[0.6598456824e+02, 0.9163596281e-02],
[0.1209819836e+02, 0.4936149294e-01],
[0.3384639924e+01, 0.1685383049e+00],
[0.1162715163e+01, 0.3705627997e+00],
[0.4515163224e+00, 0.4164915298e+00],
[0.1859593559e+00, 0.1303340841e+00]
]
# Параметри STO-6G


# Обчислення STO-6G
sto_6g = np.zeros_like(r)
for i in range(6):
    gauss =  params[i][1] * (2 * params[i][0] / np.pi) ** (3/4) * np.exp(-params[i][0] * r ** 2)
    sto_6g +=  gauss
    plt.plot(r, gauss, linestyle='--', label=f'GTO {i+1}')


# Побудова графіків
plt.plot(r, sto_1s, label='STO-1s', color='blue', linewidth=2)
plt.plot(r, sto_6g, label='STO-6G', color='red', linewidth=2)
plt.xlabel('r (Bohr)')
plt.ylabel('Функція')
plt.legend()
plt.title('Порівняння STO-1s та STO-6G')
plt.grid()
plt.savefig("STO6G.pdf")
plt.show()
