# Breast Cancer Classification: Naive Bayes vs Logistic Regression

A Python machine-learning project that compares Gaussian Naive Bayes and Logistic Regression on a binary breast-cancer classification dataset.

## Dataset

This experiment uses the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository.

- Samples: 569
- Input features: 30 numerical cell-nuclei measurements
- Classes: Malignant and Benign
- Missing values: 0

The original `wdbc.data` file has an ID column, a diagnosis column, and 30 numerical features. The ID column is removed because it does not help the prediction.

Dataset source: [UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast%2Bcancer%2Bwisconsin%2Bdiagnostic)

## Data Preparation

1. Upload and load `wdbc.data`.
2. Convert class labels: `M` to Malignant and `B` to Benign.
3. Remove the ID column.
4. Split the data into 426 training samples and 143 testing samples.
5. Standardize features with `StandardScaler`.

Standardization changes feature values to a comparable scale:

\[
z = \frac{x - \mu}{\sigma}
\]

## Models

### Gaussian Naive Bayes

Gaussian Naive Bayes calculates the probability of a sample being Malignant or Benign. It assumes that features are independent and normally distributed within each class.

### Logistic Regression

Logistic Regression assigns weights to features and calculates a probability between 0 and 1 for each class. It does not make the same feature-independence assumption as Naive Bayes.

## Test Results

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Naive Bayes | 94.4% | 95.9% | 88.7% | 92.2% |
| Logistic Regression | 96.5% | 98.0% | 92.5% | 95.1% |

Logistic Regression performed better on every metric. Its higher recall is especially important because recall measures how many actual Malignant samples are correctly identified.

## Evaluation and Visualizations

- Class-distribution bar chart
- Metric comparison chart for Accuracy, Precision, Recall, and F1-score
- Naive Bayes confusion matrix
- Logistic Regression confusion matrix

In each confusion matrix, diagonal values are correct predictions. Off-diagonal values are incorrect predictions. A False Negative means a Malignant sample was predicted as Benign, so reducing False Negatives is important.

## Conclusion

Both models gave strong results. However, Logistic Regression was the better model for this dataset because it achieved higher accuracy, precision, recall, and F1-score. The cell features are related to each other, so Logistic Regression can handle this data better than Naive Bayes, which assumes feature independence.

## Requirements

```text
pandas
matplotlib
seaborn
scikit-learn
```

## How to Run

1. Download and extract the UCI dataset.
2. Upload `wdbc.data` when the notebook asks for a file.
3. Run all notebook cells from top to bottom.
