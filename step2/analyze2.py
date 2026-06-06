
import pandas as pd
from fuzzy_inference import FuzzyInference

# بارگذاری داده تست
df_test = pd.read_csv("output/X_test_full.csv")

# بارگذاری طبقه‌بند
classifier = FuzzyInference("rules_results/ga_selected_rules.csv")

# پیش‌بینی همه نمونه‌ها
y_true = df_test['status'].values
y_pred = classifier.predict_dataset(df_test)

# پیدا کردن اندیس‌های درست و اشتباه
correct_indices = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]
wrong_indices = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]

print("=" * 70)
print("نمونه‌های درست طبقه‌بندی شده (5 نمونه اول)")
print("=" * 70)
for idx in correct_indices[:5]:
    row = df_test.iloc[idx]
    print(f"\nنمونه {idx}:")
    print(f"  واقعی: {row['status']}")
    print(f"  پیش‌بینی: {y_pred[idx]}")
    # نمایش چند ویژگی مهم
    print(f"  Tin_label: {row['Tin_label']}, Tin_degree: {row['Tin_degree']:.3f}")
    print(f"  CO2_label: {row['CO2_label']}, CO2_degree: {row['CO2_degree']:.3f}")
    print(f"  Hin_label: {row['Hin_label']}, Hin_degree: {row['Hin_degree']:.3f}")

print("\n" + "=" * 70)
print("نمونه‌های اشتباه طبقه‌بندی شده (5 نمونه اول)")
print("=" * 70)
for idx in wrong_indices[:5]:
    row = df_test.iloc[idx]
    print(f"\nنمونه {idx}:")
    print(f"  واقعی: {row['status']}")
    print(f"  پیش‌بینی: {y_pred[idx]}")
    print(f"  Tin_label: {row['Tin_label']}, Tin_degree: {row['Tin_degree']:.3f}")
    print(f"  CO2_label: {row['CO2_label']}, CO2_degree: {row['CO2_degree']:.3f}")
    print(f"  Hin_label: {row['Hin_label']}, Hin_degree: {row['Hin_degree']:.3f}")

# تحلیل دلایل احتمالی اشتباهات
print("\n" + "=" * 70)
print("تحلیل دلایل احتمالی اشتباهات")
print("=" * 70)

poor_wrong = 0
acceptable_wrong = 0
good_wrong = 0
low_confidence = 0

for idx in wrong_indices:
    row = df_test.iloc[idx]
    true_class = row['status']
    if true_class == 'Poor':
        poor_wrong += 1
    elif true_class == 'Acceptable':
        acceptable_wrong += 1
    else:
        good_wrong += 1

print(f"تعداد کل اشتباهات: {len(wrong_indices)}")
print(f"  - Poor اشتباه شده: {poor_wrong} بار")
print(f"  - Acceptable اشتباه شده: {acceptable_wrong} بار")
print(f"  - Good اشتباه شده: {good_wrong} بار")

