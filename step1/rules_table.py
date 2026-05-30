import pandas as pd

rules = []

temp_rules_list = [
    {"Antecedent": "T_in is Cold AND EnvRisk is High", "Consequent": "TempControl is StrongHeat"},
    {"Antecedent": "T_in is Cold AND EnvRisk is Medium", "Consequent": "TempControl is StrongHeat"},
    {"Antecedent": "T_in is Cold AND EnvRisk is Low", "Consequent": "TempControl is Heat"},
    {"Antecedent": "T_in is Normal", "Consequent": "TempControl is NoChange"},
    {"Antecedent": "T_in is Hot AND Light_eff is High", "Consequent": "TempControl is StrongCool"},
    {"Antecedent": "T_in is Hot AND EnvRisk is High", "Consequent": "TempControl is StrongCool"},
    {"Antecedent": "T_in is Hot AND EnvRisk is Low", "Consequent": "TempControl is Cool"},
    {"Antecedent": "T_in is Hot AND weather is sunny", "Consequent": "TempControl is StrongCool"},
    {"Antecedent": "T_in is Cold AND weather is cold", "Consequent": "TempControl is StrongHeat"},
]
for r in temp_rules_list:
    rules.append({"Category": "Temperature", **r})

hum_rules_list = [
    {"Antecedent": "H_in is Dry AND T_in is Hot", "Consequent": "HumControl is StrongHum"},
    {"Antecedent": "H_in is Dry AND T_in is Normal", "Consequent": "HumControl is Hum"},
    {"Antecedent": "H_in is Dry AND weather is dry", "Consequent": "HumControl is StrongHum"},
    {"Antecedent": "H_in is Normal", "Consequent": "HumControl is NoChange"},
    {"Antecedent": "H_in is Wet AND H_out is Wet", "Consequent": "HumControl is Dry"},
    {"Antecedent": "H_in is Wet AND weather is rainy", "Consequent": "HumControl is StrongDry"},
    {"Antecedent": "H_in is Wet AND weather is humid", "Consequent": "HumControl is StrongDry"},
    {"Antecedent": "H_in is Wet AND T_in is Cold", "Consequent": "HumControl is Dry"},
]
for r in hum_rules_list:
    rules.append({"Category": "Humidity", **r})

light_rules_list = [
    {"Antecedent": "Light_eff is Low AND N is High", "Consequent": "LightControl is High"},
    {"Antecedent": "Light_eff is Low AND N is Medium", "Consequent": "LightControl is Mid"},
    {"Antecedent": "Light_eff is Low AND N is Low", "Consequent": "LightControl is Low"},
    {"Antecedent": "Light_eff is Normal", "Consequent": "LightControl is No"},
    {"Antecedent": "Light_eff is High", "Consequent": "LightControl is No"},
    {"Antecedent": "Light_eff is Low AND weather is night", "Consequent": "LightControl is High"},
    {"Antecedent": "Light_eff is Normal AND weather is sunny", "Consequent": "LightControl is No"},
    {"Antecedent": "Light_eff is Low AND weather is cloudy", "Consequent": "LightControl is Mid"},
]
for r in light_rules_list:
    rules.append({"Category": "Light", **r})

co2_rules_list = [
    {"Antecedent": "CO2 is Low AND N is High", "Consequent": "CO2Control is IncHigh"},
    {"Antecedent": "CO2 is Low AND N is Medium", "Consequent": "CO2Control is IncLow"},
    {"Antecedent": "CO2 is Low AND Light_eff is Normal", "Consequent": "CO2Control is IncHigh"},
    {"Antecedent": "CO2 is Normal", "Consequent": "CO2Control is Zero"},
    {"Antecedent": "CO2 is High", "Consequent": "CO2Control is DecLow"},
    {"Antecedent": "CO2 is High AND T_in is Hot", "Consequent": "CO2Control is DecHigh"},
    {"Antecedent": "CO2 is High AND EnvRisk is High", "Consequent": "CO2Control is DecLow"},
    {"Antecedent": "CO2 is Low AND Light_eff is Low", "Consequent": "CO2Control is IncLow"},
    {"Antecedent": "CO2 is Low AND weather is night", "Consequent": "CO2Control is IncLow"},
    {"Antecedent": "CO2 is Normal AND N is High", "Consequent": "CO2Control is IncLow"},
]
for r in co2_rules_list:
    rules.append({"Category": "CO2", **r})

energy_rules_list = [
    {"Antecedent": "EnvRisk is High AND Light_eff is High", "Consequent": "LightControl is No", "Category": "Energy (Light)"},
    {"Antecedent": "T_in is Hot AND Light_eff is High", "Consequent": "LightControl is No", "Category": "Energy (Light)"},
    {"Antecedent": "T_in is Normal AND H_in is Wet", "Consequent": "HumControl is NoChange", "Category": "Energy (Humidity)"},
    {"Antecedent": "EnvRisk is High AND T_in is Normal", "Consequent": "TempControl is NoChange", "Category": "Energy (Temperature)"},
    {"Antecedent": "CO2 is Low AND N is High AND Light_eff is Normal", "Consequent": "CO2Control is IncHigh", "Category": "Energy (CO2)"},
    {"Antecedent": "CO2 is Low AND N is High AND Light_eff is Low", "Consequent": "CO2Control is IncLow", "Category": "Energy (CO2)"},
    {"Antecedent": "CO2 is High AND EnvRisk is High", "Consequent": "CO2Control is DecLow", "Category": "Energy (CO2)"},
]
for r in energy_rules_list:
    rules.append(r)

df_rules = pd.DataFrame(rules)
df_rules.insert(0, "Rule#", range(1, len(df_rules)+1))

print(df_rules.to_string(index=False))