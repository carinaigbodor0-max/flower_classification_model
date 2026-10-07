# Iris Dataset Multi-Class Classification

## Project Overview

This project demonstrates a simple **multi-class classification** task using the Iris dataset and Python's Scikit-learn library.

The goal is to train a machine learning model that can predict the species of an Iris flower from four measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

The model classifies each flower into one of three species:

- Setosa
- Versicolor
- Virginica

## Dataset

The project uses the built-in Iris dataset provided by `scikit-learn`.

### Features

| Feature | Description |
|---|---|
| Sepal Length | Length of the sepal in centimeters |
| Sepal Width | Width of the sepal in centimeters |
| Petal Length | Length of the petal in centimeters |
| Petal Width | Width of the petal in centimeters |

### Target Classes

| Target | Species |
|---:|---|
| 0 | Setosa |
| 1 | Versicolor |
| 2 | Virginica |

## Machine Learning Workflow

The notebook follows these main steps:

1. Load the Iris dataset.
2. Separate the features (`X`) from the target (`y`).
3. Split the data into training and testing sets.
4. Train a Logistic Regression classifier.
5. Make predictions on the test data.
6. Evaluate the model using accuracy and a classification report.
7. Visualize the results with a confusion matrix.
8. Create a prediction function for new flower measurements.

## Train-Test Split

The dataset was divided using:

```python
train_test_split(
    X, y,
    test_size=0.2,
    random_state=1
)
```

This produced:

- **120 training samples**
- **30 testing samples**

The notebook also notes that feature scaling was not required for the Iris dataset at that stage because the feature values were relatively close in range.

## Model Used

### Logistic Regression

The classifier used in the completed Iris classification workflow is:

```python
from sklearn.linear_model import LogisticRegression

clf_model = LogisticRegression(max_iter=200)
clf_model.fit(X_train, y_train)
```

Logistic Regression is used here to classify each flower into one of the three Iris species.

## Model Evaluation

The trained model was evaluated using the test set.

### Accuracy

The model achieved:

**96.67% accuracy**

```text
Accuracy: 0.9666666666666667
```

This means that the model correctly classified 29 out of the 30 test samples.

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Setosa | 1.00 | 1.00 | 1.00 | 11 |
| Versicolor | 1.00 | 0.92 | 0.96 | 13 |
| Virginica | 0.86 | 1.00 | 0.92 | 6 |
| **Accuracy** | | | **0.97** | **30** |

The notebook also generates a confusion matrix to compare the actual classes with the model's predictions.

## Making New Predictions

A prediction function was created so that new flower measurements can be passed to the trained model:

```python
def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
    sample = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred = clf_model.predict(sample)[0]
    species = iris.target_names
    prediction = f'The specie of flower is {species[pred]}'
    return prediction
```

### Example

```python
print(predict_flower(5.1, 3.5, 1.4, 0.2))
```

Output:

```text
The specie of flower is setosa
```

Another example:

```python
print(predict_flower(6.0, 3.0, 4.8, 1.8))
```

Output:

```text
The specie of flower is virginica
```

## Tools and Libraries

The notebook uses:

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

Scikit-learn components used in the notebook include:

- `load_iris`
- `train_test_split`
- `LogisticRegression`
- `accuracy_score`
- `classification_report`
- `confusion_matrix`
- `ConfusionMatrixDisplay`

The notebook also contains learning examples covering:

- Missing-value imputation with `SimpleImputer`
- `KNNImputer`
- `StandardScaler`
- `MinMaxScaler`
- `MaxAbsScaler`

These preprocessing examples are included in the notebook as demonstrations and are separate from the completed Iris classification workflow.

## Project Structure

```text
Iris-Classification/
│
├── classification_(multi_class)Iris Dataset(1).ipynb
└── README.md
```

## Key Learning Outcomes

Through this project, I practiced:

- Understanding a multi-class classification problem
- Loading a dataset with Scikit-learn
- Separating features and target variables
- Splitting data into training and testing sets
- Training a classification model
- Making predictions
- Evaluating a model using accuracy
- Reading a classification report
- Understanding a confusion matrix
- Creating a reusable prediction function
- Exploring different approaches to missing-value handling
- Exploring different feature-scaling techniques

## Conclusion

This project provided a practical introduction to **multi-class classification** using the Iris dataset.

The Logistic Regression model performed well on the test data, achieving **96.67% accuracy**. The project also provided an opportunity to practice the important stages of a basic machine learning workflow, from preparing the data to training, evaluating, and using a model for new predictions.

## Notebook

The complete implementation and experiments are available in the Jupyter Notebook included in this repository.
