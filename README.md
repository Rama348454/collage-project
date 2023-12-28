import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function to load a CSV file with error handling
def load_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse CSV file '{file_path}'.")
        return None


# Load ideal data
ideal_data = load_csv("C:/Users/G15/Desktop/Datasets1/ideal.csv")
if ideal_data is not None:
    print(ideal_data.head())
    print(ideal_data.info())


# Load train datasets
train_datasets = []
train_files = ["C:/Users/G15/Desktop/Datasets1/train.csv"]
for train_file in train_files:
    train_data = load_csv(train_file)
    if train_data is not None:
        train_datasets.append(train_data)

for train_dataset in train_datasets:
    print(train_dataset.head())
    print(train_dataset.info())


# Load test dataset
test_dataset = load_csv("C:/Users/G15/Desktop/Datasets1/test.csv")
if test_dataset is not None:
    print(test_dataset.head())
    print(test_dataset.info())


# Function to calculate the sum of squared deviations
def sum_squared_deviations(y_real, y_pred):
    return np.sum((y_real - y_pred) ** 2)


# Placeholder function for fitting an ideal function and calculating SSD
def fit_and_evaluate(data, function):
    # Fit the function to the data and return the sum of squared deviations
    pass


# Selection of the best four ideal functions
best_functions = []

for train_dataset in train_datasets:
    least_ssd = float('inf')
    best_function = None

    for ideal_function in ideal_data.columns[1:]:
        ssd = fit_and_evaluate(train_dataset, ideal_function)

        if ssd is not None and float(ssd) < float(least_ssd):
            least_ssd = ssd
            best_function = ideal_function

    best_functions.append(best_function)


# Function to map test data using best functions
def map_test_data(test_data, best_functions):
    pass


mapped_test_data = map_test_data(test_dataset, best_functions)

# Plotting ideal functions
for column in ideal_data.columns[1:]:
    plt.plot(ideal_data['x'], ideal_data[column], label=column)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Ideal Functions Plot')
plt.legend()
plt.show()

# Plotting ideal function outputs for varying x in train datasets
for column in train_datasets[0].columns[1:]:
    plt.plot(train_datasets[0]['x'], train_datasets[0][column], label=column)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Ideal Function Outputs for varying x')
plt.legend(loc='upper right')
plt.show()

# Scatter plot of test dataset
df = pd.DataFrame(test_dataset)
plt.scatter(df['x'], df['y'])
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.show()

# Unit tests using unittest module
import unittest

class TestFunctionSelection(unittest.TestCase):

    def test_sum_squared_deviations(self):
        y_real = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        self.assertEqual(sum_squared_deviations(y_real, y_pred), 0)

    # Other unit tests for data loading, function fitting, mapping, etc.

if __name__ == '__main__':
    unittest.main()
