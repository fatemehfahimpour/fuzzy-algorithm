import pandas as pd
import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import os
from step2.rule_extraction import match_rule_to_row
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from step2.rule_extraction import match_rule_to_row
from fuzzy_inference import FuzzyInference

os.makedirs("result", exist_ok=True)
def extract_features_from_antecedent(antecedent_str):
    pattern = r"(\w+)="
    return re.findall(pattern, antecedent_str)

rules_df = pd.read_csv("rules_results/ga_selected_rules.csv")

good_rules = rules_df[rules_df['consequent'] == 'Good']#سوال 2 بخش 2.11

if good_rules.empty:
    print("هیچ قانونی با نتیجه 'Good' یافت نشد.")
else:
    print(f"\nتعداد قوانین با نتیجه 'Good': {len(good_rules)}")

feature_counter = Counter()
for _, rule in good_rules.iterrows():
    antecedent = rule['antecedent']
    features = extract_features_from_antecedent(antecedent)
    feature_counter.update(features)

print("\n مهم‌ترین ویژگی‌ها در قوانین کلاس Good")
print(f"{'رتبه':<6} {'ویژگی':<12} {'تعداد تکرار':<12}")
print("-" * 30)
for i, (feature, count) in enumerate(feature_counter.most_common(), 1):
    print(f"{i:<6} {feature:<12} {count:<12}")


poor_rules = rules_df[rules_df['consequent'] == 'Poor']#سوال 3 بخش 2.11

if poor_rules.empty:
    print("هیچ قانونی با نتیجه 'Poor' یافت نشد.")
    exit()

print(f"تعداد قوانین با نتیجه 'Poor': {len(poor_rules)}")

feature_counter = Counter()
for _, rule in poor_rules.iterrows():
    antecedent = rule['antecedent']
    features = extract_features_from_antecedent(antecedent)
    feature_counter.update(features)

print("\n مهم‌ترین ویژگی‌ها در قوانین کلاس Poor")
print(f"{'رتبه':<6} {'ویژگی':<12} {'تعداد تکرار':<12}")
print("-" * 30)

for i, (feature, count) in enumerate(feature_counter.most_common(), 1):
    print(f"{i:<6} {feature:<12} {count:<12}")


class RuleSetComparator:# سوال 4 بخش 2.11
    def __init__(self, rules_before_file, rules_after_file, test_data_file):
        self.rules_before_file = rules_before_file
        self.rules_after_file = rules_after_file
        self.test_data = pd.read_csv(test_data_file)
        self.classifier_before = FuzzyInference(rules_before_file)
        self.classifier_after = FuzzyInference(rules_after_file)
        self.rules_before_df = pd.read_csv(rules_before_file)
        self.rules_after_df = pd.read_csv(rules_after_file)

    def evaluate_classifier(self, classifier, name):
        y_true = self.test_data['status'].values
        y_pred = classifier.predict_dataset(self.test_data)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=['Poor', 'Acceptable', 'Good'])

        return {
            'name': name,
            'num_rules': len(classifier.rules),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': cm
        }

    def compare(self):
        print("\n" + "-" * 60)
        print("مقایسه عملکرد قوانین قبل و بعد از الگوریتم ژنتیک")
        print("-" * 60)

        result_before = self.evaluate_classifier(self.classifier_before, "Before GA")
        result_after = self.evaluate_classifier(self.classifier_after, "After GA")

        print(f"\n{'معیار':<20} {'قبل از GA':<20} {'بعد از GA':<20}")
        print("-" * 60)
        print(f"{'تعداد قوانین':<20} {result_before['num_rules']:<20} {result_after['num_rules']:<20}")
        print(f"{'(Accuracy)':<20} {result_before['accuracy']:.4f}{'':<16} {result_after['accuracy']:.4f}")
        print(f"{'(Precision)':<20} {result_before['precision']:.4f}{'':<16} {result_after['precision']:.4f}")
        print(f"{'(Recall)':<20} {result_before['recall']:.4f}{'':<16} {result_after['recall']:.4f}")
        print(f"{'F1-score':<20} {result_before['f1_score']:.4f}{'':<16} {result_after['f1_score']:.4f}")

        acc_diff = result_after['accuracy'] - result_before['accuracy']
        if acc_diff > 0:
            print(f"\n دقت بعد از GA به میزان {acc_diff:.4f} ({acc_diff * 100:.2f}%) بهبود یافته است.")
        elif acc_diff < 0:
            print(f"\n دقت بعد از GA به میزان {abs(acc_diff):.4f} ({abs(acc_diff) * 100:.2f}%) کاهش یافته است.")
        else:
            print("\n دقت بدون تغییر باقی مانده است.")

        # نمایش ماتریس درهم‌ریختگی
        print("\nماتریس درهم‌ریختگی قبل از GA:")
        print(pd.DataFrame(result_before['confusion_matrix'],
                           index=['Poor', 'Acceptable', 'Good'],
                           columns=['Poor', 'Acceptable', 'Good']))
        print("\nماتریس درهم‌ریختگی بعد از GA:")
        print(pd.DataFrame(result_after['confusion_matrix'],
                           index=['Poor', 'Acceptable', 'Good'],
                           columns=['Poor', 'Acceptable', 'Good']))

        return result_before, result_after


class RuleAnalyzer:
    def __init__(self, rules_file, test_data_file):
        self.rules = pd.read_csv(rules_file)
        self.test_data = pd.read_csv(test_data_file)
        print(f"تعداد قوانین بارگذاری شده: {len(self.rules)}")
        print(f"تعداد نمونه‌های آزمون: {len(self.test_data)}")

    def compute_activation_frequency(self):#تعداد دفعات فعال‌سازی هر قانون
        activation_counts = []
        for idx, rule in self.rules.iterrows():
            antecedent = rule['antecedent']
            count = 0
            for _, row in self.test_data.iterrows():
                if match_rule_to_row(antecedent, row) > 0:
                    count += 1
            activation_counts.append(count)
        self.rules['activation_frequency'] = activation_counts
        return self.rules

    def top_rules_by_confidence(self, n=10):# 10 قانون با بیشترین وزن اطمینان
        return self.rules.nlargest(n, 'confidence')[['antecedent', 'consequent', 'confidence']]

    def top_rules_by_activation_frequency(self, n=10):#10 قانون با بیشترین دفعات فعال‌سازی
        return self.rules.nlargest(n, 'activation_frequency')[
            ['antecedent', 'consequent', 'confidence', 'activation_frequency']]

    def most_important_rules(self, n=10, weight_conf=0.5, weight_freq=0.5):
        max_conf = self.rules['confidence'].max()
        max_freq = self.rules['activation_frequency'].max()
        self.rules['importance_score'] = (
                weight_conf * (self.rules['confidence'] / max_conf) +
                weight_freq * (self.rules['activation_frequency'] / max_freq)
        )
        return self.rules.nlargest(n, 'importance_score')[
            ['antecedent', 'consequent', 'confidence', 'activation_frequency', 'importance_score']]

    def run_analysis(self):
        print("\n" + "-" * 60)
        print("تحلیل قوانین انتخاب‌شده پس از الگوریتم ژنتیک")
        print("-" * 60)

        # محاسبه دفعات فعال‌سازی
        self.compute_activation_frequency()

        # 1. قوانین با بیشترین وزن اطمینان
        print("\n1. قوانین با بیشترین وزن اطمینان (Top 10):")
        print(self.top_rules_by_confidence(10).to_string(index=False))

        # 2. قوانین با بیشترین دفعات فعال‌سازی روی داده آزمون
        print("\n2. قوانین با بیشترین دفعات فعال‌سازی روی داده آزمون (Top 10):")
        print(self.top_rules_by_activation_frequency(10).to_string(index=False))

        # 3. مهم‌ترین قوانین
        print("\n3. مهم‌ترین قوانین (ترکیب اطمینان و دفعات فعال‌سازی) - Top 10:")
        important = self.most_important_rules(10)
        print(important.to_string(index=False))

        self.plot_comparison()

    def plot_comparison(self):#رسم نمودار پراکندگی اطمینان در مقابل دفعات فعال‌سازی
        plt.figure(figsize=(10, 6))
        plt.scatter(self.rules['confidence'], self.rules['activation_frequency'], alpha=0.6, c='blue')
        plt.xlabel('Confidence')
        plt.ylabel('Activation Frequency')
        plt.title('Rule Analysis: Confidence vs Activation Frequency')
        plt.grid(alpha=0.3)
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/rules_scatter.png")
        plt.show()


if __name__ == "__main__":
    RULES_FILE = "rules_results/ga_selected_rules.csv"
    TEST_DATA_FILE = "output/X_test_full.csv"
    analyzer = RuleAnalyzer(RULES_FILE, TEST_DATA_FILE)
    analyzer.run_analysis()

    RULES_BEFORE = "rules_results/rules_final_rules.csv"
    RULES_AFTER = "rules_results/ga_selected_rules.csv"
    TEST_DATA = "output/X_test_full.csv"

    comparator = RuleSetComparator(RULES_BEFORE, RULES_AFTER, TEST_DATA)
    comparator.compare()