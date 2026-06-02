import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('..')
from step1.membership_function import trimf, trapmf


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"  {len(df)} رکورد, {len(df.columns)} ویژگی")
        return df
    except FileNotFoundError:
        print(f" not found")
        return None


def extract_domains_from_train_data(df):
    domains = {}

    # ویژگی‌های عددی (به جز W که دسته‌ای است)
    numerical_cols = ['T_in', 'T_out', 'H_in', 'H_out', 'L', 'Solar', 'CO2', 'Wind', 'N', 'E']

    for col in numerical_cols:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            # اضافه کردن 10% حاشیه
            margin = (max_val - min_val) * 0.1
            domain_min = min_val - margin
            domain_max = max_val + margin
            domains[col] = (domain_min, domain_max)
            print(f"  {col}: داده [{min_val:.2f}, {max_val:.2f}] → دامنه فازی [{domain_min:.2f}, {domain_max:.2f}]")

    return domains


DOMAINS = {}


def set_domains(domains):
    global DOMAINS
    DOMAINS = domains


def fuzzify_three_level(x, feature_name, left_label, mid_label, right_label):
    if feature_name not in DOMAINS:
        return {left_label: 0, mid_label: 0, right_label: 0}

    min_val, max_val = DOMAINS[feature_name]
    span = max_val - min_val

    p20 = min_val + 0.20 * span
    p35 = min_val + 0.35 * span
    p50 = min_val + 0.50 * span
    p65 = min_val + 0.65 * span
    p80 = min_val + 0.80 * span

    return {
        left_label: trapmf(x, min_val, min_val, p20, p35),
        mid_label: trimf(x, p20, p50, p80),
        right_label: trapmf(x, p65, p80, max_val, max_val)
    }


def fuzzify_Tin(x):
    return fuzzify_three_level(x, 'T_in', 'Cold', 'Normal', 'Hot')


def fuzzify_Tout(x):
    return fuzzify_three_level(x, 'T_out', 'Cold', 'Normal', 'Hot')


def fuzzify_Hin(x):
    return fuzzify_three_level(x, 'H_in', 'Dry', 'Normal', 'Wet')


def fuzzify_Hout(x):
    return fuzzify_three_level(x, 'H_out', 'Dry', 'Normal', 'Wet')


def fuzzify_L(x):
    return fuzzify_three_level(x, 'L', 'Low', 'Normal', 'High')


def fuzzify_Solar(x):
    return fuzzify_three_level(x, 'Solar', 'Low', 'Normal', 'High')


def fuzzify_CO2(x):
    return fuzzify_three_level(x, 'CO2', 'Low', 'Normal', 'High')


def fuzzify_Wind(x):
    return fuzzify_three_level(x, 'Wind', 'Low', 'Medium', 'High')


def fuzzify_N(x):
    return fuzzify_three_level(x, 'N', 'Low', 'Normal', 'High')


def fuzzify_E(x):
    return fuzzify_three_level(x, 'E', 'Low', 'Normal', 'High')


# فازی‌سازی تمام ویژگی‌های یک رکورد
def fuzzify_all_features(row):
    return {
        "Tin": fuzzify_Tin(row['T_in']),
        "Tout": fuzzify_Tout(row['T_out']),
        "Hin": fuzzify_Hin(row['H_in']),
        "Hout": fuzzify_Hout(row['H_out']),
        "L": fuzzify_L(row['L']),
        "Solar": fuzzify_Solar(row['Solar']),
        "CO2": fuzzify_CO2(row['CO2']),
        "Wind": fuzzify_Wind(row['Wind']),
        "N": fuzzify_N(row['N']),
        "E": fuzzify_E(row['E'])
    }


# برچسب با بیشترین درجه عضویت
def get_max_membership_label(fuzzy_dict):
    return max(fuzzy_dict, key=fuzzy_dict.get)


# تبدیل رکورد به برچسب‌های فازی
def fuzzify_row_to_labels(row):
    fz = fuzzify_all_features(row)
    return {feat: get_max_membership_label(fz[feat]) for feat in fz}


def save_fuzzy_labels(df, output_path):
    print(f"\n ذخیره برچسب‌های فازی در {output_path}...")

    records = []
    for idx, row in df.iterrows():
        labels = fuzzify_row_to_labels(row)

        # اضافه کردن W
        if 'W_encoded' in row:
            labels['W'] = row['W_encoded']

        # اضافه کردن status
        if 'status' in row:
            labels['status'] = row['status']

        records.append(labels)

    df_fuzzy = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_fuzzy.to_csv(output_path, index=False)
    print(f"{len(df_fuzzy)} رکورد در {output_path} ذخیره شد.")
    return df_fuzzy


def save_full_fuzzy_values(df, output_path):
    print(f"\n ذخیره مقادیر کامل فازی در {output_path}...")

    records = []

    for _, row in df.iterrows():
        fz = fuzzify_all_features(row)
        record = {}

        for feat, memberships in fz.items():
            # ذخیره همه درجات عضویت
            for label, degree in memberships.items():
                record[f"{feat}_{label}"] = degree

            # ذخیره برچسب برنده
            winner_label = get_max_membership_label(memberships)
            winner_degree = memberships[winner_label]

            record[f"{feat}_label"] = winner_label
            record[f"{feat}_degree"] = winner_degree

        # اضافه کردن W
        if 'W_encoded' in row:
            record['W'] = row['W_encoded']
        elif 'W' in row:
            record['W'] = row['W']

        # اضافه کردن status
        if 'status' in row:
            record['status'] = row['status']

        records.append(record)

    df_fuzzy_full = pd.DataFrame(records)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_fuzzy_full.to_csv(output_path, index=False)

    print(f"{len(df_fuzzy_full)} رکورد در {output_path} ذخیره شد.")
    return df_fuzzy_full


# رسم توابع عضویت
def plot_all_membership_functions(output_dir="membership_plots"):
    os.makedirs(output_dir, exist_ok=True)

    features = {
        "T_in": (fuzzify_Tin, ["Cold", "Normal", "Hot"]),
        "T_out": (fuzzify_Tout, ["Cold", "Normal", "Hot"]),
        "H_in": (fuzzify_Hin, ["Dry", "Normal", "Wet"]),
        "H_out": (fuzzify_Hout, ["Dry", "Normal", "Wet"]),
        "L": (fuzzify_L, ["Low", "Normal", "High"]),
        "Solar": (fuzzify_Solar, ["Low", "Normal", "High"]),
        "CO2": (fuzzify_CO2, ["Low", "Normal", "High"]),
        "Wind": (fuzzify_Wind, ["Low", "Medium", "High"]),
        "N": (fuzzify_N, ["Low", "Normal", "High"]),
        "E": (fuzzify_E, ["Low", "Normal", "High"])
    }

    for name, (func, labels) in features.items():
        if name not in DOMAINS:
            print(f"   {name}: دامنه پیدا نشد")
            continue

        min_val, max_val = DOMAINS[name]
        x = np.linspace(min_val, max_val, 500)

        plt.figure(figsize=(10, 5))
        for label in labels:
            y = [func(xi)[label] for xi in x]
            plt.plot(x, y, label=label, linewidth=2)

        plt.xlabel(name)
        plt.ylabel("Degree of Membership")
        plt.title(f"Membership Functions - {name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/{name}_mf.png", dpi=150, bbox_inches='tight')
        plt.close()


# ================================================
# نمایش نمونه
def demo_fuzzification(sample_row):
    print("\n" + "=" * 70)
    print(" نمونه فازی‌سازی یک رکورد:")
    print("=" * 70)

    print("\n مقادیر عددی ورودی:")
    for col in ['T_in', 'T_out', 'H_in', 'H_out', 'L', 'Solar', 'CO2', 'Wind', 'N', 'E']:
        if col in sample_row:
            print(f"  {col}: {sample_row[col]}")

    fz = fuzzify_all_features(sample_row)

    print("\n برچسب فازی Winner:")
    for feat, values in fz.items():
        winner = get_max_membership_label(values)
        mu = values[winner]
        print(f"  {feat}: {winner} (μ = {mu:.3f})")

    print("=" * 70)


def main():
    # 1. بارگذاری داده آموزش
    print("\n  بارگذاری داده آموزش...")
    df_train = load_data("output/X_train.csv")
    if df_train is None:
        return

    # 2. بارگذاری داده تست
    print("\n  بارگذاری داده تست...")
    df_test = load_data("output/X_test.csv")

    # 3. استخراج دامنه (فقط از داده آموزش)
    print("\n  استخراج دامنه از داده آموزش...")
    domains = extract_domains_from_train_data(df_train)
    set_domains(domains)

    # 4. رسم توابع عضویت
    plot_all_membership_functions()
    demo_fuzzification(df_train.iloc[0])

    # 6. ذخیره برچسب‌های فازی برای داده آموزش
    save_fuzzy_labels(df_train, "output/X_train_fuzzy.csv")

    # 7. ذخیره برچسب‌های فازی برای داده تست
    if df_test is not None:
        print("\n  ذخیره برچسب‌های فازی تست...")
        save_fuzzy_labels(df_test, "output/X_test_fuzzy.csv")

    save_full_fuzzy_values(df_train, "output/X_train_full_fuzzy.csv")
    if df_test is not None:
        save_full_fuzzy_values(df_test, "output/X_test_full.csv")


if __name__ == "__main__":
    main()
