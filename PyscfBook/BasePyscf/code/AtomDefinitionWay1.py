# Один атом
mol = gto.M(atom='Ne 0 0 0', basis='def2-svp')

# Можна використовувати атомний номер
mol = gto.M(atom='10 0 0 0', basis='def2-svp')  # 10 = Ne
