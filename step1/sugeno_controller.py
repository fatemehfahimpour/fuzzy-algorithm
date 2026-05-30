from step1.fuzzy_controller import *

TEMP_SUGENO = {
    "StrongCool": -90,
    "Cool": -50,
    "NoChange": 0,
    "Heat": 50,
    "StrongHeat": 90,
}

HUM_SUGENO = {
    "StrongDry": -90,
    "Dry": -50,
    "NoChange": 0,
    "Hum": 50,
    "StrongHum": 90,
}

LIGHT_SUGENO = {
    "No": 15,
    "Low": 35,
    "Mid": 60,
    "High": 90,
}


CO2_SUGENO = {
    "DecHigh": -90,
    "DecLow": -40,
    "Zero": 0,
    "IncLow": 40,
    "IncHigh": 90,
}

def sugeno_defuzzify(rules, label_to_value):
    num = 0.0
    den = 0.0
    for w, label in rules:
        if w <= 0:
            continue
        z = label_to_value[label]
        num += w * z
        den += w
    return 0.0 if den == 0 else num / den


def temp_hum_light_co2_sugeno_controller(data):
    weather = str(data["W"]).lower()
    L_eff = data["L"] + data["Solar"]
    env = compute_envrisk(data["T_in"], data["T_out"], data["Wind"], weather)

    temp_fz = fuzzify_temp(data["T_in"])
    hum_in_fz = fuzzify_hum(data["H_in"])
    hum_out_fz = fuzzify_hum(data["H_out"])
    light_fz = fuzzify_light(L_eff)
    env_fz = fuzzify_env(env)
    co2_fz = fuzzify_co2(data["CO2"])
    density_fz = fuzzify_density(data["N"])

    temp_rule = temp_rules(temp_fz, env_fz, light_fz, weather)
    hum_rule = hum_rules(hum_in_fz, temp_fz, weather, hum_out_fz)
    light_rule = light_rules(light_fz, density_fz, weather)
    co2_rule = co2_rules(co2_fz, density_fz, temp_fz, weather, light_fz, env_fz)
    energy_rules = energy_efficiency_rules(co2_fz, density_fz, light_fz, env_fz, temp_fz, hum_in_fz)

    temp_rule.extend(energy_rules["temp"])
    hum_rule.extend(energy_rules["hum"])
    light_rule.extend(energy_rules["light"])
    co2_rule.extend(energy_rules["co2"])

    temp = sugeno_defuzzify(temp_rule, TEMP_SUGENO)
    hum = sugeno_defuzzify(hum_rule, HUM_SUGENO)
    light = sugeno_defuzzify(light_rule, LIGHT_SUGENO)
    co2 = sugeno_defuzzify(co2_rule, CO2_SUGENO)

    return {
        "TempControl": float(np.clip(temp, -100, 100)),
        "HumControl": float(np.clip(hum, -100, 100)),
        "LightControl": float(np.clip(light, 0, 100)),
        "CO2Control": float(np.clip(co2, -100, 100)),
    }
