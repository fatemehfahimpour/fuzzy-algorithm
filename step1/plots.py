import matplotlib.pyplot as plt
import os

class Plotter:
    @staticmethod
    def save_all_plots(df_results, output_dir="plots"):
        os.makedirs(output_dir, exist_ok=True)
        days = df_results["day"].values

        # نمودار دما
        plt.figure()
        plt.plot(days, df_results["T_in"], 'r-o')
        plt.axhline(y=22, color='k', linestyle='--', label="نرمال (22°C)")
        plt.xlabel("روز")
        plt.ylabel("دما (°C)")
        plt.title("تغییرات دمای داخل گلخانه")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/temp_inside.png")
        plt.close()

        # نمودار رطوبت
        plt.figure()
        plt.plot(days, df_results["H_in"], 'b-o')
        plt.axhline(y=55, color='k', linestyle='--', label="نرمال (55%)")
        plt.xlabel("روز")
        plt.ylabel("رطوبت (%)")
        plt.title("تغییرات رطوبت داخل گلخانه")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/hum_inside.png")
        plt.close()

        # نمودار نور مصنوعی
        plt.figure()
        plt.plot(days, df_results["L"], 'g-o')
        plt.xlabel("روز")
        plt.ylabel("شدت نور (Lux)")
        plt.title("تغییرات نور مصنوعی")
        plt.grid(True)
        plt.savefig(f"{output_dir}/light_inside.png")
        plt.close()

        # نمودار CO2
        plt.figure()
        plt.plot(days, df_results["CO2"], 'm-o')
        plt.axhline(y=800, color='k', linestyle='--', label="مقدار اولیه")
        plt.xlabel("روز")
        plt.ylabel("CO₂ (ppm)")
        plt.title("تغییرات غلظت CO₂")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/co2_inside.png")
        plt.close()

        # نمودار سیگنال‌های کنترلی
        plt.figure(figsize=(10, 6))
        plt.plot(days, df_results["TempControl"], 'r--', label="کنترل دما")
        plt.plot(days, df_results["HumControl"], 'b--', label="کنترل رطوبت")
        plt.plot(days, df_results["LightControl"], 'g--', label="کنترل نور")
        plt.plot(days, df_results["CO2Control"], 'm--', label="کنترل CO₂")
        plt.xlabel("روز")
        plt.ylabel("مقدار کنترل")
        plt.title("سیگنال‌های کنترلی خروجی از سیستم فازی")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/control_signals.png")
        plt.close()


        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(days, df_results["T_in"], 'r-o')
        axes[0, 0].set_title("دمای داخل")
        axes[0, 1].plot(days, df_results["H_in"], 'b-o')
        axes[0, 1].set_title("رطوبت داخل")
        axes[1, 0].plot(days, df_results["L"], 'g-o')
        axes[1, 0].set_title("نور مصنوعی")
        axes[1, 1].plot(days, df_results["CO2"], 'm-o')
        axes[1, 1].set_title("CO₂ داخل")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/all_in_one.png")
        plt.close()