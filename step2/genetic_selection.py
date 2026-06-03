import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
from step2.rule_extraction import match_rule_to_row

RULES_FILE = "rules_results/rules_final_rules.csv"
DATA_FILE_TEST = "output/X_test_full.csv"

ALPHA = 0.01
MIN_RULES = 10

POP_SIZE = 50
N_GENERATION = 40
CXPB = 0.8
MUTPB = 0.1
TOURN_SIZE = 3
EPS = 1e-12

FUZZY_FEATURES = ['Tin', 'Tout', 'Hin', 'Hout', 'L', 'Solar', 'CO2', 'Wind', 'N', 'E']
CATEGORICAL_FEATURES = ['W']

rules_df = pd.read_csv(RULES_FILE)
data_df_test = pd.read_csv(DATA_FILE_TEST)
SUBSET_SIZE = 50  # تعداد نمونه‌هایی که برای fitness استفاده می‌کنیم
subset_idx = np.random.choice(
    len(data_df_test),
    size=min(SUBSET_SIZE, len(data_df_test)),
    replace=False
)
subset_df = data_df_test.iloc[subset_idx]
subset_y = subset_df['status'].values

N_RULES = len(rules_df)

print("تعداد قوانین اولیه:", N_RULES)
print("تعداد نمونه‌ها:", len(data_df_test))


def predict_sample(row, selected_rules):
    class_scores = {}

    for _, rule in selected_rules.iterrows():
        antecedent = rule['antecedent']
        consequent = rule['consequent']
        confidence = rule['confidence']

        m = match_rule_to_row(antecedent, row)

        B = m * confidence

        class_scores[consequent] = class_scores.get(consequent, 0) + B

    if len(class_scores) == 0:
        return rules_df['consequent'].mode()[0]

    return max(class_scores, key=class_scores.get)


def predict_dataset(df, selected_rules):
    preds = []

    for _, row in df.iterrows():
        p = predict_sample(row, selected_rules)

        preds.append(p)

    return np.array(preds)


# Accuracy
def compute_accuracy(selected_rules):
    y_pred = predict_dataset(subset_df, selected_rules)
    return accuracy_score(subset_y, y_pred)


# Fitness Function
def fitness_function(individual):
    selected_idx = [i for i, b in enumerate(individual) if b == 1]

    if len(selected_idx) < MIN_RULES:
        return 0.0,

    selected_rules = rules_df.iloc[selected_idx]

    acc = compute_accuracy(selected_rules)

    penalty = ALPHA * (len(selected_idx) / N_RULES)

    fitness = acc - penalty

    return fitness,


# GA----------------------------------------------------------------------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)  # تعریف ژن

toolbox.register(  # ساخت کروموزوم کامل
    "individual",
    tools.initRepeat,
    creator.Individual,
    toolbox.attr_bool,
    n=N_RULES
)

toolbox.register(
    "population",
    tools.initRepeat,
    list,
    toolbox.individual
)

toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxTwoPoint)  # Two Point Crossover
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=TOURN_SIZE)  # تورنومنت سلکشن


# اجرایGA-----------------------------------------------------
def run_ga():
    pop = toolbox.population(n=POP_SIZE)

    hof = tools.HallOfFame(1)  # بهترین افراد کل الگوریتم را نگه دار

    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=CXPB,
        mutpb=MUTPB,
        ngen=N_GENERATION,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    best = hof[0]

    selected_idx = [i for i, b in enumerate(best) if b == 1]

    best_rules = rules_df.iloc[selected_idx]

    print("\n========== نتیجه نهایی GA ==========")
    print("تعداد قوانین انتخاب شده:", len(best_rules))

    # accuracy روی subset
    acc = compute_accuracy(best_rules)
    print("Subset Accuracy:", acc)

    # accuracy واقعی روی کل test
    final_pred = predict_dataset(data_df_test, best_rules)
    final_acc = accuracy_score(data_df_test['status'], final_pred)
    print("Final Test Accuracy:", final_acc)

    os.makedirs("rules_results", exist_ok=True)

    best_rules.to_csv(
        "rules_results/ga_selected_rules.csv",
        index=False
    )

    top10 = best_rules.sort_values(
        "confidence",
        ascending=False
    ).head(10)

    top10.to_csv(
        "rules_results/ga_top10_rules.csv",
        index=False
    )

    print("\n10 قانون با بیشترین confidence:")

    print(top10[['antecedent', 'consequent', 'confidence']])

    # -----------------رسم نمودار همگرایی GA----------------
    generations = log.select('gen')
    avg_fitness = log.select('avg')
    max_fitness = log.select('max')

    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_fitness, 'b-', label='میانگین برازندگی (Average)')
    plt.plot(generations, max_fitness, 'r-', label='بهترین برازندگی (Max)')
    plt.xlabel('نسل (Generation)')
    plt.ylabel('برازندگی (Fitness)')
    plt.title('همگرایی الگوریتم ژنتیک (GA Convergence)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/ga_convergence.png", dpi=150, bbox_inches='tight')
    plt.show()

    return best_rules


if __name__ == "__main__":
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    best_rules = run_ga()
