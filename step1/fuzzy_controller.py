from step1.mamdani_engine import defuzzify
from step1.membership_function import trapmf, trimf


# FUZZIFICATION-----------------------------------------------------------------
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


def fuzzify_hum(x):
    return {
        "Dry": trapmf(x, 10, 10, 30, 45),
        "Normal": trimf(x, 40, 55, 70),
        "Wet": trapmf(x, 60, 75, 90, 90),
    }


def fuzzify_env(x):
    return {
        "Low": trapmf(x, 0, 0, 8, 15),
        "Medium": trimf(x, 10, 20, 30),
        "High": trapmf(x, 25, 35, 60, 60)
    }


# defuzzified outputs--------------------------------------------------------------------
def temp_output(label, z):
    if label == "StrongCool":
        return trapmf(z, -100, -100, -70, -40)
    elif label == "Cool":
        return trimf(z, -60, -30, 0)
    elif label == "NoChange":
        return trimf(z, -10, 0, 10)
    elif label == "Heat":
        return trimf(z, 0, 30, 60)
    elif label == "StrongHeat":
        return trapmf(z, 40, 70, 100, 100)
    return 0


def hum_output(label, z):
    if label == "StrongDry":
        return trapmf(z, -100, -100, -70, -40)
    elif label == "Dry":
        return trimf(z, -60, -30, 0)
    elif label == "NoChange":
        return trimf(z, -10, 0, 10)
    elif label == "Hum":
        return trimf(z, 0, 30, 60)
    elif label == "StrongHum":
        return trapmf(z, 40, 70, 100, 100)
    return 0


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


# RULES----------------------------------------------------------------------------------------
def temp_rules(temp_in, env_risk, light_eff, weather):
    rules = []

    rules.append((min(temp_in["Cold"], env_risk["High"]), "StrongHeat"))
    rules.append((min(temp_in["Cold"], env_risk["Medium"]), "StrongHeat"))
    rules.append((min(temp_in["Cold"], env_risk["Low"]), "Heat"))
    rules.append((temp_in["Normal"], "NoChange"))
    rules.append((min(temp_in["Hot"], light_eff["High"]), "StrongCool"))
    rules.append((min(temp_in["Hot"], env_risk["High"]), "StrongCool"))
    rules.append((min(temp_in["Hot"], env_risk["Low"]), "Cool"))
    if weather == "sunny":
        rules.append((temp_in["Hot"], "StrongCool"))
    if weather == "cold":
        rules.append((temp_in["Cold"], "StrongHeat"))
    return rules


def hum_rules(hum_in, temp_in, weather, hum_out):
    rules = []

    rules.append((min(hum_in["Dry"], temp_in["Hot"]), "StrongHum"))
    rules.append((min(hum_in["Dry"], temp_in["Normal"]), "Hum"))
    if weather == "dry":
        rules.append((hum_in["Dry"], "StrongHum"))
    rules.append((hum_in["Normal"], "NoChange"))
    rules.append((min(hum_in["Wet"], hum_out["Wet"]), "Dry"))
    if weather == "rainy":
        rules.append((hum_in["Wet"], "StrongDry"))
    if weather == "humid":
        rules.append((hum_in["Wet"], "StrongDry"))
    rules.append((min(hum_in["Wet"], temp_in["Cold"]), "Dry"))
    return rules


def energy_efficiency_rules(co2_fz, density_fz, light_fz, env_fz, temp_fz, hum_fz):
    rules = {
        "light": [],
        "temp": [],
        "hum": [],
        "co2": []
    }

    rules["light"].append((min(env_fz["High"], light_fz["High"]), "No"))
    rules["light"].append((min(temp_fz["Hot"], light_fz["High"]), "No"))
    rules["hum"].append((min(temp_fz["Normal"], hum_fz["Wet"]), "NoChange"))
    rules["temp"].append((min(env_fz["High"], temp_fz["Normal"]), "NoChange"))
    rules["co2"].append((min(co2_fz["Low"], density_fz["High"], light_fz["Normal"]), "IncHigh"))
    rules["co2"].append((min(co2_fz["Low"], density_fz["High"], light_fz["Low"]), "IncLow"))
    rules["co2"].append((min(co2_fz["High"], env_fz["High"]), "DecLow"))
    return rules


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
# CONTROLLER---------------------------------------------------------------------------------
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


def temp_hum_light_co2_controller(data):
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

    temp_rule = temp_rules(temp_fz , env_fz , light_fz, weather)
    hum_rule = hum_rules(hum_in_fz,temp_fz, weather, hum_out_fz)
    light_rule = light_rules(light_fz, density_fz, weather)
    co2_rule = co2_rules(co2_fz, density_fz, temp_fz, weather, light_fz, env_fz)
    energy_rules = energy_efficiency_rules(co2_fz, density_fz, light_fz, env_fz, temp_fz, hum_in_fz)

    temp_rule.extend(energy_rules["temp"])
    hum_rule.extend(energy_rules["hum"])
    light_rule.extend(energy_rules["light"])
    co2_rule.extend(energy_rules["co2"])

    return {
        "TempControl": defuzzify(temp_rule , -100, 100, temp_output),
        "HumControl": defuzzify(hum_rule, -100, 100, hum_output),
        "LightControl": defuzzify(light_rule, 0, 100, light_output),
        "CO2Control": defuzzify(co2_rule, -100, 100, co2_output)
    }


#TEST----------------------------------------------------------------------------------
if __name__ == "__main__":
    sample = {
        "L": 500,
        "Solar": 400,
        "CO2": 1000,
        "N": 15,
        "T_in": 22,
        "T_out": 18,
        "Wind": 2,
        "W": "sunny",
        "H_in": 20,
        "H_out": 70
    }

    print(temp_hum_light_co2_controller(sample))
