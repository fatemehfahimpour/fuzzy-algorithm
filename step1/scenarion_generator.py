import pandas as pd

class ScenarioGenerator:
    @staticmethod
    def generate():
        return pd.DataFrame({
            "day": range(1, 11),
            "W": ["sunny", "sunny", "cloudy", "rainy", "stormy",
                  "cold", "cold", "humid", "dry", "night"],
            "T_out": [32, 38, 22, 18, 16, 8, 3, 28, 26, 20],
            "H_out": [45, 30, 55, 85, 75, 60, 70, 90, 20, 65],
            "Solar": [900, 1000, 300, 80, 50, 200, 100, 600, 700, 0],
            "Wind": [3, 2, 4, 6, 12, 5, 8, 2, 5, 4],
            "N": [120, 120, 150, 60, 90, 40, 110, 130, 30, 100]
        })