import pandas as pd
import numpy as np
from rule_extraction import match_rule_to_row

class FuzzyInference:
    def __init__(self, rules_file, default_class='Acceptable'):
        self.rules = pd.read_csv(rules_file)
        self.default_class = default_class
        print(f"تعداد قوانین بارگذاری شده: {len(self.rules)}")


    def predict_sample_with_scores(self, row):
        class_scores = {cls: 0.0 for cls in ['Poor', 'Acceptable', 'Good']}
        active_rules = []

        for idx, rule in self.rules.iterrows():
            antecedent = rule['antecedent']
            consequent = rule['consequent']
            confidence = rule['confidence']

            alpha = match_rule_to_row(antecedent, row)
            alpha = alpha * confidence
            class_scores[consequent] += alpha

            if alpha > 0:
                active_rules.append({
                    'rule_index': idx,
                    'antecedent': antecedent,
                    'consequent': consequent,
                    'activation': alpha
                })

        if max(class_scores.values()) == 0:
            return self.default_class, class_scores, active_rules
        predicted_class = max(class_scores, key=class_scores.get)
        return predicted_class, class_scores, active_rules


    def predict_sample(self, row):
        pred, _, _ = self.predict_sample_with_scores(row)
        return pred


    def predict_dataset(self, df):
        predictions = []
        for _, row in df.iterrows():
            pred = self.predict_sample(row)
            predictions.append(pred)
        return np.array(predictions)


    def evaluate(self, df_test, target_col='status'):
        y_true = df_test[target_col].values
        y_pred = self.predict_dataset(df_test)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=['Poor', 'Acceptable', 'Good'])

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
        return metrics


    def show_scores_for_samples(self, df_test, sample_indices, target_col='status'):
        print("\nنمایش امتیاز کلاس‌ها برای نمونه‌های آزمون")
        for idx in sample_indices:
            row = df_test.iloc[idx]
            true_class = row[target_col]
            pred_class, scores, _ = self.predict_sample_with_scores(row)
            print(f"\nنمونه {idx+1} (واقعی: {true_class}) -> پیش‌بینی {pred_class}")
            for cls, scr in scores.items():
                print(f"   امتیاز {cls}: {scr:.6f}")


    def show_active_rules_for_samples(self, df_test, sample_indices, top_n=5):
        print("\nنمایش قوانین فعال برای نمونه‌های آزمون")
        for idx in sample_indices:
            row = df_test.iloc[idx]
            _, _, active_rules = self.predict_sample_with_scores(row)
            print(f"\nنمونه {idx+1} - قوانین فعال ({len(active_rules)} قانون):")
            active_rules.sort(key=lambda x: x['activation'], reverse=True)
            for rule in active_rules[:top_n]:
                print(f"   قانون {rule['rule_index']}: {rule['antecedent']} → {rule['consequent']} (فعال‌سازی = {rule['activation']:.6f})")


if __name__ == "__main__":
    RULES_FILE = "rules_results/ga_selected_rules.csv"
    TEST_DATA_FILE = "output/X_test_full.csv"
    TRAIN_DATA_FILE = "output/X_train_full_fuzzy.csv"

    df_train = pd.read_csv(TRAIN_DATA_FILE)
    most_frequent_class = df_train['status'].mode()[0]
    print(f"کلاس پرتکرار در داده آموزش: {most_frequent_class}")

    df_test = pd.read_csv(TEST_DATA_FILE)
    classifier = FuzzyInference(RULES_FILE, default_class=most_frequent_class)

    results = classifier.evaluate(df_test)
    print("\nنتایج ارزیابی")
    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")
    print(f"F1-score : {results['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(pd.DataFrame(results['confusion_matrix'],
                       index=['Poor', 'Acceptable', 'Good'],
                       columns=['Poor', 'Acceptable', 'Good']))

    # نمایش امتیاز هر کلاس برای 5 نمونه
    sample_indices_scores = list(range(min(5, len(df_test))))
    classifier.show_scores_for_samples(df_test, sample_indices_scores)

    # نمایش قوانین فعال برای 3 نمونه
    sample_indices_rules = list(range(min(3, len(df_test))))
    classifier.show_active_rules_for_samples(df_test, sample_indices_rules)