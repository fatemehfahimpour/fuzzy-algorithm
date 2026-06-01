import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

DATA_FILE = "../greenhouse_dataset_with_status.csv"


def load_data(file_path=DATA_FILE):
    df = pd.read_csv(file_path)
    print(f" {df.shape[0]} ردیف, {df.shape[1]} ستون")
    return df


def generate_data_report(df):
    print("\n" + "=" * 50)
    print("گزارش وضعیت داده‌ها")
    print("=" * 50)
    print(f"تعداد کل رکوردها: {len(df)}")
    print(f"تعداد کل ویژگی‌ها: {len(df.columns)}")
    print(f"تعداد کلاس‌های هدف: {df['status'].nunique()}")
    print("\nتوزیع کلاس‌ها:")
    print(df['status'].value_counts())
    print("=" * 50)


#  مقادیر گمشده در هر ستون
def show_missing_per_column(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("\n" + "=" * 50)
    print("مقادیر گمشده در هر ستون")
    print("=" * 50)
    if len(missing) > 0:
        for col, val in missing.items():
            print(f"  {col}: {val} مقدار گمشده")
    else:
        print("هیچ مقدار گمشده‌ای وجود ندارد.")
    print("=" * 50)


#  مقادیر گمشده با روش جایگزینی
def check_and_handle_missing(df):
    total_missing = df.isnull().sum().sum()
    if total_missing == 0:
        print("هیچ مقدار گمشده‌ای وجود ندارد.")
        return df

    print(f" تعداد کل مقادیر گمشده: {total_missing}")

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].mean(), inplace=True)  #جایگزینی با مفادیر عددی میانگین
                print(f"  - {col}: پر شد با میانگین ({df[col].mean():.2f})")
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)  #جایگزینی با مد
                print(f"  - {col}: پر شد با مد ({df[col].mode()[0]})")
    return df



#  داده‌های پرت
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound


def report_outliers_details(df, numerical_cols):
    has_outlier = False
    for col in numerical_cols:
        if col in df.columns:
            n, low, high = detect_outliers_iqr(df, col)
            if n > 0:
                has_outlier = True
                print(f"  {col}: {n} داده پرت (محدوده نرمال: [{low:.2f}, {high:.2f}])")
            else:
                print(f"  {col}: بدون داده پرت")
    if not has_outlier:
        print("هیچ داده پرتی شناسایی نشد.")



#  W (Label Encoding)
def encode_weather(df):
    df = df.copy()
    le = LabelEncoder()
    df['W_encoded'] = le.fit_transform(df['W'])
    for i, cls in enumerate(le.classes_):
        print(f"  {cls} -> {i}")
    return df, le


#  تقسیم داده (Stratified)
def split_data_stratified(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\n تقسیم داده: {len(X_train)} آموزش, {len(X_test)} آزمون")
    return X_train, X_test, y_train, y_test


# 7) توزیع کلاس‌ها قبل و بعد از تقسیم
def show_class_distribution_before(y):
    print("\n" + "=" * 50)
    print("توزیع کلاس‌ها در کل دیتاست (قبل از تقسیم)")
    print("=" * 50)
    print(y.value_counts())
    print(f"جمع کل: {len(y)}")
    print("=" * 50)


def show_class_distribution_after(y_train, y_test):
    print("\n" + "=" * 50)
    print("توزیع کلاس‌ها بعد از تقسیم")
    print("=" * 50)
    print("\n داده آموزش (Train):")
    print(y_train.value_counts())
    print(f"جمع آموزش: {len(y_train)}")
    print("\nداده آزمون (Test):")
    print(y_test.value_counts())
    print(f"جمع آزمون: {len(y_test)}")

    # بررسی نسبت‌ها
    print("\n نسبت کلاس‌ها:")
    print(f"{'کلاس':<12} {'آموزش':<12} {'آزمون':<12}")
    print("-" * 36)
    for cls in ['Poor', 'Acceptable', 'Good']:
        train_ratio = (y_train == cls).sum() / len(y_train) * 100
        test_ratio = (y_test == cls).sum() / len(y_test) * 100
        print(f"{cls:<12} {train_ratio:.1f}%{'':<8} {test_ratio:.1f}%")
    print("=" * 50)



def save_split_data(X_train, X_test, y_train, y_test, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

if __name__ == "__main__":
    df_original = load_data()
    df = df_original.copy()

    numerical_cols = ['T_in', 'T_out', 'H_in', 'H_out', 'L', 'Solar', 'CO2', 'Wind', 'N', 'E']
    numerical_cols = [col for col in numerical_cols if col in df.columns]

    generate_data_report(df)
    show_missing_per_column(df)
    df = check_and_handle_missing(df)
    report_outliers_details(df, numerical_cols)
    df, le = encode_weather(df)
    X = df.drop(columns=['status'])
    y = df['status']
    show_class_distribution_before(y)
    X_train, X_test, y_train, y_test = split_data_stratified(X, y)
    show_class_distribution_after(y_train, y_test)
    save_split_data(X_train, X_test, y_train, y_test)

