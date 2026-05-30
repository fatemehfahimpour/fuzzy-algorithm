import numpy as np
import pandas as pd
from fuzzy_controller import temp_hum_light_co2_controller
from day_night_policy import apply_day_night_policy

class GreenhouseSimulator:
    T_RANGE = (10, 35)
    H_RANGE = (10, 90)
    L_RANGE = (0, 1900)
    CO2_RANGE = (300, 1700)

    ALPHA_T = 0.05
    BETA_T = 0.03
    ALPHA_H = 0.06
    BETA_H = 0.01
    GAMMA_H = 0.02
    ALPHA_L = 5.0
    BETA_L = 0.1
    ALPHA_C = 4.0
    BETA_C = 0.8
    GAMMA_C = 1.5

    def __init__(self, initial_state=None):
        if initial_state is None:
            self.state = {
                "T_in": 30.0,
                "H_in": 30.0,
                "L": 500.0,
                "CO2": 600.0
            }
        else:
            self.state = initial_state.copy()

    def _update(self, controls, external):
        T_in = self.state["T_in"]
        H_in = self.state["H_in"]
        L = self.state["L"]
        CO2 = self.state["CO2"]

        TempControl = controls["TempControl"]
        HumControl = controls["HumControl"]
        LightControl = controls["LightControl"]
        CO2Control = controls["CO2Control"]

        T_out = external["T_out"]
        H_out = external["H_out"]
        Solar = external["Solar"]
        Wind = external["Wind"]
        N = external["N"]

        eps_T = np.random.normal(0, 0.5)
        eps_H = np.random.normal(0, 0.5)
        eps_C = np.random.normal(0, 5)

        new_T = T_in + self.ALPHA_T * TempControl - self.BETA_T * (T_in - T_out) + eps_T
        new_H = H_in + self.ALPHA_H * HumControl + self.BETA_H * (H_out - H_in) - self.GAMMA_H * Wind + eps_H
        new_L = L + self.ALPHA_L * LightControl - self.BETA_L * Solar
        new_C = CO2 + self.ALPHA_C * CO2Control - self.BETA_C * N - self.GAMMA_C * Wind + eps_C

        self.state["T_in"] = new_T
        self.state["H_in"] = new_H
        self.state["L"] = np.clip(new_L,0,1000)
        self.state["CO2"] = np.clip(new_C,300,1700)

    def run(self, scenario_df):
        results = []
        for _, row in scenario_df.iterrows():#پیمایش هر روز
            inputs = {
                "T_in": self.state["T_in"],
                "T_out": row["T_out"],
                "H_in": self.state["H_in"],
                "H_out": row["H_out"],
                "L": self.state["L"],
                "Solar": row["Solar"],
                "CO2": self.state["CO2"],
                "Wind": row["Wind"],
                "N": row["N"],
                "W": row["W"]
            }

            controls = temp_hum_light_co2_controller(inputs)

            light_policy, co2_policy = apply_day_night_policy(
                row["W"].lower(),
                controls["LightControl"],
                controls["CO2Control"]
            )
            controls["LightControl"] = light_policy
            controls["CO2Control"] = co2_policy

            results.append({# ذخیره اطلاعات روز جاری
                "day": row["day"],
                "weather": row["W"],
                "T_in": self.state["T_in"],
                "H_in": self.state["H_in"],
                "L": self.state["L"],
                "CO2": self.state["CO2"],
                "TempControl": controls["TempControl"],
                "HumControl": controls["HumControl"],
                "LightControl": controls["LightControl"],
                "CO2Control": controls["CO2Control"],
                "T_out": row["T_out"],
                "H_out": row["H_out"],
                "Solar": row["Solar"],
                "Wind": row["Wind"],
                "N": row["N"]
            })

            # به‌روزرسانی وضعیت برای روز بعد
            external = {
                "T_out": row["T_out"],
                "H_out": row["H_out"],
                "Solar": row["Solar"],
                "Wind": row["Wind"],
                "N": row["N"]
            }
            self._update(controls, external)

        return pd.DataFrame(results)