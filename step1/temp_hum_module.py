import numpy as np

from step1.light_co2_module import compute_envrisk, fuzzify_light, fuzzify_env, fuzzify_co2, fuzzify_density
from step1.mamdani_engine import defuzzify
from step1.membership_function import trapmf, trimf
import config


def fuzzify_temp(x):
    return {
        "Cold": trapmf(x, *config.T_COLD_PARAMS),
        "Normal": trimf(x, *config.T_NORMAL_PARAMS),
        "Hot": trapmf(x, *config.T_HOT_PARAMS)
    }


def fuzzify_hum(x):
    return {
        "Dry": trapmf(x, *config.H_DRY_PARAMS),
        "Normal": trimf(x, *config.H_NORMAL_PARAMS),
        "Wet": trapmf(x, *config.H_WET_PARAMS),
    }


def temp_output(label, z):
    if label == "StrongCool":
        return trapmf(z, *config.TEMP_STRONG_COOL)
    elif label == "Cool":
        return trimf(z, *config.TEMP_COOL)
    elif label == "NoChange":
        return trimf(z, *config.TEMP_NOCHANGE)
    elif label == "Heat":
        return trimf(z, *config.TEMP_HEAT)
    elif label == "StrongHeat":
        return trapmf(z, *config.TEMP_STRONG_HEAT)
    return 0


def hum_output(label, z):
    if label == "StrongDry":
        return trapmf(z, *config.HUM_STRONG_DRY)
    elif label == "Dry":
        return trimf(z, *config.HUM_DRY)
    elif label == "NoChange":
        return trimf(z, *config.HUM_NOCHANGE)
    elif label == "Hum":
        return trimf(z, *config.HUM_HUMIDIFY)
    elif label == "StrongHum":
        return trapmf(z, *config.HUM_STRONG_HUM)
    return 0


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
    rules.append((min(hum_in["Wet"], hum_out["High"]), "Dry"))
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


def temp_hum_controller(data):
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
    energy_rules = energy_efficiency_rules(co2_fz, density_fz, light_fz, env_fz, temp_fz, hum_in_fz)

    temp_rule.extend(energy_rules["temp"])
    hum_rule.extend(energy_rules["hum"])

    return {
        "TempControl": defuzzify(temp_rule , -100, 100, temp_output),
        "hum_control": defuzzify(hum_rule, -100, 100, hum_output)
    }
