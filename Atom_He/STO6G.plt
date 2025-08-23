set terminal pdf
set output "STO6Ggp.pdf"

# Определение параметров
zeta = 1.7
a1 = 0.6598456824e+2
a2 = 0.1209819836e+2
a3 = 0.3384639924e+1
a4 = 0.1162715163e+1
a5 = 0.4515163224e+0
a6 = 0.1859593559e+0

C1 = 0.9163596281e-2
C2 = 0.4936149294e-1
C3 = 0.1685383049e+0
C4 = 0.3705627997e+0
C5 = 0.4164915298e+0
C6 = 0.1303340841e+0

# Определение функций
STO(r) = (1/(4*pi))**0.5 * (2*zeta)**1.5 / sqrt(2) * exp(-zeta*r)

GTO(a, C, r) = C * (2*a/pi)**(3./4) * exp(-a*r**2)

STO6G(r) = GTO(a1, C1, r) + GTO(a2, C2, r) + GTO(a3, C3, r) + GTO(a4, C4, r) + GTO(a5, C5, r) + GTO(a6, C6, r)

# Построение графиков
set xrange [0:3]
set yrange [0:1.3]
set xlabel "r"
set ylabel "Value"
set title "STO and STO6G Functions"
set grid

plot STO(x) title "STO" with lines linecolor rgb "red", \
     STO6G(x) title "STO6G" with lines linecolor rgb "blue", \
     GTO(a1, C1, x) title "GTO(α1, C1)" with lines linecolor rgb "cyan", \
     GTO(a2, C2, x) title "GTO(α2, C2)" with lines linecolor rgb "green", \
     GTO(a3, C3, x) title "GTO(α3, C3)" with lines linecolor rgb "brown", \
     GTO(a4, C4, x) title "GTO(α4, C4)" with lines linecolor rgb "orange", \
     GTO(a5, C5, x) title "GTO(α5, C5)" with lines linecolor rgb "magenta", \
     GTO(a6, C6, x) title "GTO(α6, C6)" with lines linecolor rgb "#00ee00"