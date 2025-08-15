

## **Code Description – Car Price Prediction Using Machine Learning and Deep Learning**

This script implements two regression models—a traditional **Linear Regression** model and a **Deep Learning** model—to predict car prices from a dataset. The workflow follows a standard **machine learning pipeline**: data loading, preprocessing, model training, evaluation, and visualization.

---

### **1. Libraries Used**

* **Data Handling**:
  `numpy`, `pandas` – for numerical operations and dataset manipulation.
* **Visualization**:
  `matplotlib`, `seaborn` – for scatter plots, histograms, and residual distribution plots.
* **Machine Learning (ML)**:
  `sklearn` – for train-test split, preprocessing, model training, and evaluation metrics.
* **Deep Learning (DL)**:
  `tensorflow.keras` – for building and training the neural network.

---

### **2. Data Loading & Preprocessing**

```python
carData = pd.read_csv('CarPrice.csv')
carData = carData.drop('CarName', axis=1)
```

* Loads the dataset from a CSV file.
* Removes the `CarName` column, which is non-numerical and not needed for prediction.

```python
carData = pd.get_dummies(carData, columns=[...], drop_first=True)
```

* Converts categorical variables (fuel type, body style, engine type, etc.) into numerical format using **one-hot encoding**.
* `drop_first=True` avoids dummy variable trap by removing the first category from each encoded feature.

---

### **3. Feature Selection & Data Splitting**

```python
X = carData.drop('price', axis=1)
y = carData['price']
X_train, X_test, y_train, y_test = train_test_split(...)
```

* **Features (X)**: All columns except `price`.
* **Target (y)**: Car price.
* Dataset is split into **80% training** and **20% testing** subsets for unbiased evaluation.

---

### **4. Feature Scaling**

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

* **Standardizes** features to have zero mean and unit variance, which improves convergence for both ML and DL models.

---

### **5. Linear Regression Model**

```python
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

* Trains a **Linear Regression** model.
* Makes predictions on the test set.

```python
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

* **Mean Squared Error (MSE)**: Measures prediction error magnitude.
* **R² score**: Measures how well the model explains variance in the target variable.

---

### **6. Visualizations for Linear Regression**

* **Actual vs. Predicted Prices** – checks correlation between true and predicted values.
* **Horsepower vs. Price** – explores a single feature relationship.
* **Residual Distribution** – evaluates whether residuals are normally distributed.

---

### **7. Deep Learning Model**

```python
dl_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
```

* A **feedforward neural network** with:

  * **Three hidden layers** (128, 64, 32 neurons).
  * **ReLU activation** for non-linearity.
  * **Dropout layers** to reduce overfitting (20% of neurons dropped randomly per pass).
  * **Single neuron output layer** for regression.

```python
dl_model.compile(optimizer='adam', loss='mean_squared_error')
dl_model.fit(X_train, y_train, epochs=100, batch_size=32)
```

* Uses **Adam optimizer** for efficient gradient descent.
* **Mean Squared Error** as the loss function.
* Trains for **100 epochs** with a batch size of **32**.

---

### **8. Deep Learning Model Evaluation**

```python
y_dl_model = dl_model.predict(X_test).flatten()
mse_dl_model = mean_squared_error(y_test, y_dl_model)
r2_dl_model = r2_score(y_test, y_dl_model)
```

* Predictions are flattened into a 1D array.
* Computes **MSE** and **R²** to compare with the Linear Regression model.

---

### **9. Visualization for Deep Learning Model**

* **Actual vs. Predicted Prices** – visualizes performance for the DL model, similar to the ML model.

---

### **10. Summary**

This code demonstrates:

1. **Data preprocessing** using pandas and one-hot encoding.
2. **Feature scaling** to improve model performance.
3. **Training and evaluating** both a traditional **Linear Regression** model and a **Deep Learning** model.
4. **Comparative analysis** using metrics and visualizations.

Both approaches aim to predict car prices, but the deep learning model’s non-linear nature may better capture complex feature relationships, while linear regression provides simplicity and interpretability.

---
