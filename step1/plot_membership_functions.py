import numpy as np
import matplotlib.pyplot as plt
import os
from fuzzy_controller import (
    fuzzify_temp, fuzzify_hum, fuzzify_light, fuzzify_co2,
    fuzzify_density, fuzzify_env,
    temp_output, hum_output, light_output, co2_output)


def plot_mf(x_range, func_dict, title, xlabel, ylabel="Degree of membership", save_path=None):
    x = np.linspace(x_range[0], x_range[1], 500)
    plt.figure(figsize=(8, 4))
    for label, func in func_dict.items():
        y = [func(xi) for xi in x]
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    output_dir = "membership_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Tin
    plot_mf(
        x_range=(10, 35),
        func_dict={
            "Cold": lambda x: fuzzify_temp(x)["Cold"],
            "Normal": lambda x: fuzzify_temp(x)["Normal"],
            "Hot": lambda x: fuzzify_temp(x)["Hot"]
        },
        title="Membership Functions - Temperature Inside (T_in)",
        xlabel="Temperature (°C)",
        save_path=f"{output_dir}/temp_input_mf.png"
    )

    # Hin
    plot_mf(
        x_range=(10, 90),
        func_dict={
            "Dry": lambda x: fuzzify_hum(x)["Dry"],
            "Normal": lambda x: fuzzify_hum(x)["Normal"],
            "Wet": lambda x: fuzzify_hum(x)["Wet"]
        },
        title="Membership Functions - Humidity Inside (H_in)",
        xlabel="Humidity (%)",
        save_path=f"{output_dir}/hum_input_mf.png"
    )

    # light
    plot_mf(
        x_range=(0, 1900),
        func_dict={
            "Low": lambda x: fuzzify_light(x)["Low"],
            "Normal": lambda x: fuzzify_light(x)["Normal"],
            "High": lambda x: fuzzify_light(x)["High"]
        },
        title="Membership Functions - Effective Light (L_eff)",
        xlabel="Light intensity (Lux)",
        save_path=f"{output_dir}/light_input_mf.png"
    )

    # co2
    plot_mf(
        x_range=(300, 1700),
        func_dict={
            "Low": lambda x: fuzzify_co2(x)["Low"],
            "Normal": lambda x: fuzzify_co2(x)["Normal"],
            "High": lambda x: fuzzify_co2(x)["High"]
        },
        title="Membership Functions - CO2 Concentration",
        xlabel="CO2 (ppm)",
        save_path=f"{output_dir}/co2_input_mf.png"
    )

    # N
    plot_mf(
        x_range=(1, 30),
        func_dict={
            "Low": lambda x: fuzzify_density(x)["Low"],
            "Medium": lambda x: fuzzify_density(x)["Medium"],
            "High": lambda x: fuzzify_density(x)["High"]
        },
        title="Membership Functions - Plant Density (N)",
        xlabel="Density (plants/area)",
        save_path=f"{output_dir}/density_input_mf.png"
    )

    # فشار محیطی
    plot_mf(
        x_range=(0, 60),
        func_dict={
            "Low": lambda x: fuzzify_env(x)["Low"],
            "Medium": lambda x: fuzzify_env(x)["Medium"],
            "High": lambda x: fuzzify_env(x)["High"]
        },
        title="Membership Functions - Environmental Risk",
        xlabel="EnvRisk",
        save_path=f"{output_dir}/env_input_mf.png"
    )

    # TempControl
    plot_mf(
        x_range=(-100, 100),
        func_dict={
            "StrongCool": lambda z: temp_output("StrongCool", z),
            "Cool": lambda z: temp_output("Cool", z),
            "NoChange": lambda z: temp_output("NoChange", z),
            "Heat": lambda z: temp_output("Heat", z),
            "StrongHeat": lambda z: temp_output("StrongHeat", z)
        },
        title="Membership Functions - Temperature Control Output",
        xlabel="TempControl",
        save_path=f"{output_dir}/temp_output_mf.png"
    )

    # HumControl
    plot_mf(
        x_range=(-100, 100),
        func_dict={
            "StrongDry": lambda z: hum_output("StrongDry", z),
            "Dry": lambda z: hum_output("Dry", z),
            "NoChange": lambda z: hum_output("NoChange", z),
            "Hum": lambda z: hum_output("Hum", z),
            "StrongHum": lambda z: hum_output("StrongHum", z)
        },
        title="Membership Functions - Humidity Control Output",
        xlabel="HumControl",
        save_path=f"{output_dir}/hum_output_mf.png"
    )

    # LightControl
    plot_mf(
        x_range=(0, 100),
        func_dict={
            "No": lambda z: light_output("No", z),
            "Low": lambda z: light_output("Low", z),
            "Mid": lambda z: light_output("Mid", z),
            "High": lambda z: light_output("High", z)
        },
        title="Membership Functions - Light Control Output",
        xlabel="LightControl",
        save_path=f"{output_dir}/light_output_mf.png"
    )

    # CO2Control
    plot_mf(
        x_range=(-100, 100),
        func_dict={
            "DecHigh": lambda z: co2_output("DecHigh", z),
            "DecLow": lambda z: co2_output("DecLow", z),
            "Zero": lambda z: co2_output("Zero", z),
            "IncLow": lambda z: co2_output("IncLow", z),
            "IncHigh": lambda z: co2_output("IncHigh", z)
        },
        title="Membership Functions - CO2 Control Output",
        xlabel="CO2Control",
        save_path=f"{output_dir}/co2_output_mf.png"
    )


if __name__ == "__main__":
    main()
