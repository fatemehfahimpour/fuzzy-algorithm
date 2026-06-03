import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzy_inference import FuzzyInference
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import os
import sys
from step2.rule_extraction import match_rule_to_row
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class plots:
    def __init__(self, classifier, rules_before_ga=None, rules_after_ga=None):
        self.classifier = classifier
        self.rules_before = rules_before_ga
        self.rules_after = rules_after_ga


    def compute_metrics(self, df_test, target_col='status'):
        y_true = df_test[target_col].values
        y_pred = self.classifier.predict_dataset(df_test)
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'classification_report': classification_report(y_true, y_pred, zero_division=0),
            'y_true': y_true,
            'y_pred': y_pred
        }
        return metrics


    def plot_confusion_matrix(self, df_test, target_col='status', save_path=None):
        y_true = df_test[target_col].values
        y_pred = self.classifier.predict_dataset(df_test)
        cm = confusion_matrix(y_true, y_pred, labels=['Poor', 'Acceptable', 'Good'])

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Poor', 'Acceptable', 'Good'],
                    yticklabels=['Poor', 'Acceptable', 'Good'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - Fuzzy Classifier')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
        plt.show()
        return cm


    def plot_rule_count_comparison(self, save_path=None):
        if self.rules_before is None or self.rules_after is None:
            print("داده‌ای از قوانین قبل/بعد از GA موجود نیست.")
            return

        counts = [len(self.rules_before), len(self.rules_after)]
        labels = ['Before GA', 'After GA']

        plt.figure(figsize=(5, 4))
        bars = plt.bar(labels, counts, color=['skyblue', 'salmon'])
        plt.ylabel('Number of Rules')
        plt.title('Rule Count Comparison (GA Selection)')
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(count), ha='center', va='bottom')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
        plt.show()


    def plot_rule_activation_for_samples(self, df_test, sample_indices, top_n=10, save_path=None):
        all_activations = []
        rule_indices = set()
        for idx in sample_indices:
            row = df_test.iloc[idx]
            _, _, active_rules = self.classifier.predict_sample_with_scores(row)
            act_dict = {r['rule_index']: r['activation'] for r in active_rules}
            all_activations.append(act_dict)
            for r in active_rules:
                rule_indices.add(r['rule_index'])

        # مرتب‌سازی قوانین بر اساس بیشترین فعال‌سازی در بین نمونه‌ها
        rule_indices = list(rule_indices)
        max_act = {}
        for r_idx in rule_indices:
            max_act[r_idx] = max(act.get(r_idx, 0) for act in all_activations)
        top_rules = sorted(rule_indices, key=lambda x: max_act[x], reverse=True)[:top_n]

        # ساخت ماتریس فعال‌سازی
        n_samples = len(sample_indices)
        n_rules = len(top_rules)
        matrix = np.zeros((n_samples, n_rules))
        for i, act_dict in enumerate(all_activations):
            for j, r_idx in enumerate(top_rules):
                matrix[i, j] = act_dict.get(r_idx, 0)

        # رسم نقشه حرارتی
        fig, ax = plt.subplots(figsize=(max(8, n_rules * 0.5), max(5, n_samples * 0.6)))
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(np.arange(n_rules))
        ax.set_yticks(np.arange(n_samples))
        ax.set_xticklabels([f"R{r}" for r in top_rules], rotation=45, ha='right')
        ax.set_yticklabels([f"Sample {idx}" for idx in sample_indices])
        # نوشتن مقادیر روی سلول‌ها
        for i in range(n_samples):
            for j in range(n_rules):
                text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                               ha="center", va="center",
                               color="black" if matrix[i, j] < 0.5 else "white")
        plt.colorbar(im, label='Activation Degree')
        plt.title('Rule Activation for Selected Test Samples')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
        plt.show()


    def show_misclassified_samples(self, df_test, target_col='status', n=5, save_path=None):
        y_true = df_test[target_col].values
        y_pred = self.classifier.predict_dataset(df_test)
        correct_indices = np.where(y_true == y_pred)[0]
        wrong_indices = np.where(y_true != y_pred)[0]

        print("\n=== 5 نمونه درست طبقه‌بندی شده ===")
        for i in correct_indices[:n]:
            print(f"  Index {i}: True={y_true[i]}, Pred={y_pred[i]}")
        print("\n=== 5 نمونه اشتباه طبقه‌بندی شده ===")
        for i in wrong_indices[:n]:
            print(f"  Index {i}: True={y_true[i]}, Pred={y_pred[i]}")
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("نمونه‌های درست طبقه‌بندی شده:\n")
                for i in correct_indices[:n]:
                    f.write(f"Index {i}: True={y_true[i]}, Pred={y_pred[i]}\n")
                f.write("\nنمونه‌های اشتباه طبقه‌بندی شده:\n")
                for i in wrong_indices[:n]:
                    f.write(f"Index {i}: True={y_true[i]}, Pred={y_pred[i]}\n")


# ----------------نمودار فراوانی کلاس‌های واقعی و پیش‌بینی شده------------------
def plot_class_distribution_comparison(y_true, y_pred, save_path="plots/class_distribution.png"):
    classes = ['Poor', 'Acceptable', 'Good']
    true_counts = [np.sum(y_true == c) for c in classes]
    pred_counts = [np.sum(y_pred == c) for c in classes]
    x = np.arange(len(classes))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, true_counts, width, label='True', color='skyblue')
    plt.bar(x + width/2, pred_counts, width, label='Predicted', color='salmon')
    plt.xlabel('کلاس')
    plt.ylabel('تعداد نمونه')
    plt.title('مقایسه فراوانی کلاس‌های واقعی و پیش‌بینی شده')
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    from fuzzy_inference import FuzzyInference

    RULES_FILE = "rules_results/ga_selected_rules.csv"
    TEST_DATA_FILE = "output/X_test_full.csv"
    TRAIN_DATA_FILE = "output/X_train_full_fuzzy.csv"

    df_train = pd.read_csv(TRAIN_DATA_FILE)
    most_frequent = df_train['status'].mode()[0]
    df_test = pd.read_csv(TEST_DATA_FILE)
    classifier = FuzzyInference(RULES_FILE, default_class=most_frequent)

    y_pred = classifier.predict_dataset(df_test)
    y_true = df_test['status'].values

    plot_class_distribution_comparison(y_true, y_pred)

    evaluator = plots(classifier, rules_before_ga=pd.read_csv("rules_results/rules_final_rules.csv"),
                      rules_after_ga=pd.read_csv("rules_results/ga_selected_rules.csv"))

    #-----------------نمودار فعال‌سازی قوانین برای 3 نمونه------------------
    sample_indices = [0, 10, 20]# سه نمونه دلخواه
    evaluator.plot_rule_activation_for_samples(df_test, sample_indices, top_n=10,save_path="plots/rule_activation_heatmap.png")

    # رسم ماتریس درهم‌ریختگی
    evaluator.plot_confusion_matrix(df_test, save_path="plots/confusion_matrix.png")

    # مقایسه تعداد قوانین قبل و بعد از GA
    evaluator.plot_rule_count_comparison(save_path="plots/rule_count_comparison.png")

    # محاسبه معیارها
    metrics = evaluator.compute_metrics(df_test)
    print("Accuracy:", metrics['accuracy'])
    print(metrics['classification_report'])