import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


files = [
    "runs/test/PriceAndWeightDistribution_best_1.csv",
    "runs/test/PriceAndWeightDistribution_best_0.5.csv",
    "runs/test/PriceAndWeightDistribution_best_0.75.csv",
    "runs/test/PriceAndWeightDistribution_best_0.375.csv",
]

def read_weights(file):
    data = pd.read_csv(file)
    weights = np.array(data)[:, 1:5]
    return weights

weights_list = [read_weights(file) for file in files]

average_weights = [weights.mean(axis=0) for weights in weights_list]
standard_errors = [weights.std(axis=0) / np.sqrt(weights.shape[0]) for weights in weights_list]

labels_Col = ['0.375', '0.5', '0.75', '1']
labels_Title = ['AAPL', 'CL=F', 'TSLA', 'Cash']

n_columns = len(average_weights[0])
colors = plt.cm.rainbow(np.linspace(0, 1, n_columns))

for col in range(n_columns):
    col_values = [row[col] for row in average_weights]
    col_errors = [row[col] for row in standard_errors]
    plt.bar(labels_Col, col_values, yerr=col_errors, color=['red', 'green', 'blue', 'yellow'], capsize=5)
    plt.title(f"Invested in {labels_Title[col]}")
    plt.xlabel("Agents")
    plt.ylabel("Average Weights")
    plt.ylim(0, 0.4)
    plt.show()