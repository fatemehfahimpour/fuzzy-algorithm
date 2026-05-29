def apply_day_night_policy(weather, light_control, co2_control):
    if weather == "night":
        # شب: افزایش نور مصنوعی حداکثر 30% و تزریق CO2 حداکثر 20%
        light_control = min(light_control, 30)
        co2_control = min(co2_control, 20)
    elif weather == "sunny":
        # روز آفتابی: افزایش نور مصنوعی حداکثر 10% (صرفه‌جویی)
        light_control = min(light_control, 10)
    return light_control, co2_control