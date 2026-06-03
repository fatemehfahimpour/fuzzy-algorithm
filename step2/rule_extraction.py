import os

import numpy as np
import pandas as pd

EPS = 1e-12
FUZZY_FEATURES = ['T_in', 'T_out', 'H_in', 'H_out', 'L', 'Solar', 'CO2', 'Wind', 'N', 'E']
CATEGORICAL_FEATURES = ['W']


def build_antecedent(row):
    parts = []
    # ویژگی‌های فازی
    for feat in FUZZY_FEATURES:
        label_col = f"{feat}_label"
        if label_col in row.index and pd.notna(row[label_col]):
            parts.append(f"{feat}={row[label_col]}")

    # ویژگی categorical مثل W
    for feat in CATEGORICAL_FEATURES:
        if feat in row.index and pd.notna(row[feat]):
            parts.append(f"{feat}={row[feat]}")

    return " AND ".join(parts)


def calculate_weight(row, method='min'):
    degrees = []
    for feat in FUZZY_FEATURES:
        feature_max_degree_col = f"{feat}_degree"
        if feature_max_degree_col in row.index and pd.notna(row[feature_max_degree_col]):
            degrees.append(float(row[feature_max_degree_col]))

    for feat in CATEGORICAL_FEATURES:
        if feat in row.index and pd.notna(row[feat]):
            degrees.append(1.0)

    if not degrees:
        return 0.0

    if method == "product":
        w = 1.0
        for d in degrees:
            w *= d
        return float(w)
    else:
        return float(min(degrees))


def generate_raw_rules(df_train, target_col='status', weight_method='min'):
    rules = []
    for idx, raw in df_train.iterrows():
        antecedent = build_antecedent(raw)
        consequence = raw[target_col]
        weight = calculate_weight(raw, method=weight_method)
        rules.append({
            "antecedent": antecedent,
            "consequent": consequence,
            'weight': weight,
            'source_row': idx
        })

    return pd.DataFrame(rules)


def aggregate_rules(raw_rules_df):
    grouped = (
        # گروه بندی بر اساس مقدم و نتیجه
        # همه قوانینی که مقدم و نتیجه یکسان دارند در یک گروه قرار میگیرند
        raw_rules_df
        .groupby(['antecedent', 'consequent'], as_index=False)
        .agg(
            weight=('weight', 'max'),  # انتخاب بیشترین وزن برای هر گروه
            count=('weight', 'size')  # تعداد قوانین هر گروه
        )
    )
    return grouped


def parse_antecedent(antecedent_str):
    if pd.isna(antecedent_str) or antecedent_str.strip() == "":
        return []
    return [part.strip() for part in antecedent_str.split(" AND ")]


def match_rule_to_row(rule_antecedent, row):  # match(Rj ,xi)
    terms = parse_antecedent(rule_antecedent)
    if not terms:
        return 0.0

    match_vals = []

    for term in terms:
        if "=" not in term:
            continue

        feat, label = term.split("=", 1)
        feat = feat.strip()
        label = label.strip()

        # ویژگی فازی پیوسته
        if feat in FUZZY_FEATURES:
            # چک کردن اینکه آیا لیبل ویژگی مد نظر در سطر هست یا نه
            # به عنوان مثال tin = hot پس لیبل برابر hot است
            # ایا tin در این سطر hot هست یا نه
            deg_col = f"{feat}_{label}"
            if deg_col in row.index and pd.notna(row[deg_col]):
                match_vals.append(float(row[deg_col]))
            else:
                match_vals.append(0.0)

        # ویژگی categorical
        elif feat in CATEGORICAL_FEATURES:
            # اگر antecedent نوشته W=3
            try:
                row_val = str(int(row[feat]))
            except:
                row_val = str(row[feat])

            match_vals.append(1.0 if row_val == label else 0.0)

        else:
            match_vals.append(0.0)

    if not match_vals:
        return 0.0

    # اینجا با product حساب شده
    m = 1.0
    for v in match_vals:
        m *= v
    return float(m)


def compute_confidence_for_rules(rules_df, df_train, target_col='status'):
    confidences = []

    for _, rule in rules_df.iterrows():
        antecedent = rule['antecedent']
        consequent = rule['consequent']

        numerator = 0.0
        denominator = 0.0

        for _, row in df_train.iterrows():
            m = match_rule_to_row(antecedent, row)

            denominator += m
            if row[target_col] == consequent:
                numerator += m

        confidence = numerator / (denominator + EPS)
        confidences.append(confidence)

    rules_df = rules_df.copy()
    rules_df['confidence'] = confidences
    return rules_df


def resolve_conflicts(rules_df):
    resolved = (
        rules_df  # مرتب کردن قوانین بر اساس مقدار مقدم و اطمینان و وزن خود قانون
        .sort_values(['antecedent', 'confidence', 'weight'],
                     ascending=[True, False, False])  # اطمینان و وزن به صورت نزولی مرتب شده اند
        .drop_duplicates(subset=['antecedent'],
                         keep='first')  # حذف کردن قانون هایی که مقدم یکسان دارند و نگه داشتن بهترین بر اساس اطمینان و وزن
        .reset_index(drop=True)  # شماره گذاری مجدد
    )
    return resolved


# باید درست بشه این بخش وارد نشده است در کدها
def apply_dont_care(rule_antecedent, max_ratio=0.3):
    """
    این تابع فعلاً فقط اسکلت است.
    در نسخه ساده می‌توان بعداً روی antecedentها اعمالش کرد.
    """
    terms = parse_antecedent(rule_antecedent)
    if not terms:
        return rule_antecedent

    n = len(terms)
    max_dc = int(np.floor(max_ratio * n))
    # اینجا اگر بخواهی می‌توانی بعضی ویژگی‌ها را با # جایگزین کنی
    # فعلاً فعال نشده تا خروجی شفاف بماند
    return rule_antecedent


def wang_mendel_rule_extraction(df_train, target_col='status', weight_method='min', save_prefix='rules'):
    # قوانین خام
    raw_rules = generate_raw_rules(df_train, target_col=target_col, weight_method=weight_method)

    # حذف تکراری‌ها با مقدم و نتیجه یکسان
    aggregated_rules = aggregate_rules(raw_rules)

    # محاسبه confidence
    aggregated_rules = compute_confidence_for_rules(aggregated_rules, df_train, target_col=target_col)

    # resolve conflicts
    final_rules = resolve_conflicts(aggregated_rules)

    # os.makedirs('rules_results', exist_ok=True)
    # ذخیره فایل‌ها
    raw_rules.to_csv(f'rules_results/{save_prefix}_raw_rules.csv', index=False, encoding='utf-8-sig')
    aggregated_rules.to_csv(f'rules_results/{save_prefix}_aggregated_rules.csv', index=False, encoding='utf-8-sig')
    final_rules.to_csv(f'rules_results/{save_prefix}_final_rules.csv', index=False, encoding='utf-8-sig')

    # گزارش
    print("=== Wang-Mendel Rule Extraction Summary ===")
    print(f"Raw rules count: {len(raw_rules)}")
    print(f"Aggregated rules count: {len(aggregated_rules)}")
    print(f"Final rules count after conflict resolution: {len(final_rules)}")

    return raw_rules, aggregated_rules, final_rules


if __name__ == "__main__":
    file_path = os.path.join("output", "X_train_full_fuzzy.csv")
    df_train = pd.read_csv(file_path)
    wang_mendel_rule_extraction(df_train)
