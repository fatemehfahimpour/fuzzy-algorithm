from scenarion_generator import ScenarioGenerator
from greenhouse_simulator import GreenhouseSimulator
from plots import Plotter

def main():
    scenario = ScenarioGenerator.generate()
    simulator = GreenhouseSimulator()
    results_df = simulator.run(scenario)

    results_df.to_csv("simulation_results.csv", index=False)

    Plotter.save_all_plots(results_df)

    print("نمونه نتایج (۵ روز اول):")
    print(results_df[["TempControl", "HumControl", "LightControl", "CO2Control"]].head())

if __name__ == "__main__":
    main()