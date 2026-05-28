import numpy as np


#روش مرکز جسم
def defuzzify(rules, zmin, zmax, output_func, step=1):
    # zmin , zmax :  بازه‌ی عددی خروجی قطعی
    num = 0 #صورت کسر
    den = 0 #مخرج کسر

    for z in np.arange(zmin, zmax + step, step):
        mu = 0
        for strength, label in rules:
            mu = max(mu, min(strength, output_func(label, z)))
        num += z * mu
        den += mu

    return num / den if den != 0 else 0