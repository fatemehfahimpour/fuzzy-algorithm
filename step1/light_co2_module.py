import numpy as np
from membership_function import trimf, trapmf


def compute_envrisk(T_in, T_out, Wind, W):
    k_map = {
        "sunny": 0.9,
        "cloudy": 1.0,
        "rainy": 1.15,
        "stormy": 1.25,
        "cold": 1.1,
        "humid": 1.05,
        "dry": 1.0,
        "night": 1.0
    }

    k = k_map.get(str(W).lower(), 1.0)

    delta_T = abs(T_in - T_out)
    env = k * (delta_T + 0.5 * Wind)

    return max(0, min(60, env))


#  FUZZIFICATION
def fuzzify_light(x):
    return {
        "Low": trapmf(x, 0, 0, 300, 500),
        "Normal": trimf(x, 400, 700, 1000),
        "High": trapmf(x, 900, 1200, 1900, 1900)
    }


def fuzzify_co2(x):
    return {
        "Low": trapmf(x, 300, 300, 600, 800),
        "Normal": trimf(x, 700, 1000, 1300),
        "High": trapmf(x, 1200, 1400, 1700, 1700)
    }


def fuzzify_density(x):
    return {
        "Low": trapmf(x, 1, 1, 6, 10),
        "Medium": trimf(x, 8, 15, 22),
        "High": trapmf(x, 20, 25, 30, 30)
    }


def fuzzify_temp(x):
    return {
        "Cold": trapmf(x, 10, 10, 16, 20),
        "Normal": trimf(x, 18, 22, 26),
        "Hot": trapmf(x, 24, 28, 35, 35)
    }


def fuzzify_env(x):
    return {
        "Low": trapmf(x, 0, 0, 8, 15),
        "Medium": trimf(x, 10, 20, 30),
        "High": trapmf(x, 25, 35, 60, 60)
    }


def light_output(label, z):
    if label == "No":
        return trapmf(z, 0, 0, 10, 25)
    if label == "Low":
        return trimf(z, 15, 35, 55)
    if label == "Mid":
        return trimf(z, 40, 60, 80)
    if label == "High":
        return trapmf(z, 70, 85, 100, 100)
    return 0


def co2_output(label, z):
    if label == "DecHigh":
        return trapmf(z, -100, -100, -70, -40)
    if label == "DecLow":
        return trimf(z, -60, -30, 0)
    if label == "Zero":
        return trimf(z, -15, 0, 15)
    if label == "IncLow":
        return trimf(z, 0, 30, 60)
    if label == "IncHigh":
        return trapmf(z, 40, 70, 100, 100)
    return 0


def light_rules(light, density, weather):
    rules = []

    rules.append((min(light["Low"], density["High"]), "High"))
    rules.append((min(light["Low"], density["Medium"]), "Mid"))
    rules.append((min(light["Low"], density["Low"]), "Low"))
    rules.append((light["Normal"], "No"))
    rules.append((light["High"], "No"))
    if weather == "night":
        rules.append((light["Low"], "High"))
    if weather == "sunny":
        rules.append((light["Normal"], "No"))
    if weather == "cloudy":
        rules.append((light["Low"], "Mid"))
    return rules


def co2_rules(co2, density, temp, weather, light, env):
    rules = []
    rules.append((min(co2["Low"], density["High"]), "IncHigh"))
    rules.append((min(co2["Low"], density["Medium"]), "IncLow"))
    rules.append((min(co2["Low"], light["Normal"]), "IncHigh"))
    rules.append((co2["Normal"], "Zero"))
    rules.append((co2["High"], "DecLow"))
    rules.append((min(co2["High"], temp["Hot"]), "DecHigh"))
    rules.append((min(co2["High"], env["High"]), "DecLow"))
    rules.append((min(co2["Low"], light["Low"]), "IncLow"))
    if weather == "night":
        rules.append((co2["Low"], "IncLow"))
    rules.append((min(co2["Normal"], density["High"]), "IncLow"))

    return rules


def defuzzify(rules, zmin, zmax, mf):
    step = 1
    num = 0
    den = 0

    for z in np.arange(zmin, zmax + step, step):
        mu = 0
        for strength, label in rules:
            mu = max(mu, min(strength, mf(label, z)))

        num += z * mu
        den += mu

    return num / den if den != 0 else 0


def light_co2_controller(data):
    W = str(data["W"]).lower()

    L_eff = data["L"] + data["Solar"]

    env = compute_envrisk(data["T_in"], data["T_out"], data["Wind"], W)

    light_fz = fuzzify_light(L_eff)
    co2_fz = fuzzify_co2(data["CO2"])
    density_fz = fuzzify_density(data["N"])
    temp_fz = fuzzify_temp(data["T_in"])
    env_fz = fuzzify_env(env)

    light_r = light_rules(light_fz, density_fz, W)
    co2_r = co2_rules(co2_fz, density_fz, temp_fz, W, light_fz, env_fz)

    return {
        "LightControl": defuzzify(light_r, 0, 100, light_output),
        "CO2Control": defuzzify(co2_r, -100, 100, co2_output)
    }


if __name__ == "__main__":
    sample = {
        "L": 500,
        "Solar": 400,
        "CO2": 1000,
        "N": 15,
        "T_in": 22,
        "T_out": 18,
        "Wind": 2,
        "W": "sunny"
    }

    print(light_co2_controller(sample))
