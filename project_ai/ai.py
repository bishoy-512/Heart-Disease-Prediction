import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# --- Load and preprocess data ---
df = pd.read_csv('heart.csv', low_memory=False)
df['chol'] = pd.to_numeric(df['chol'], errors='coerce')

for column in df.columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mean())

def remove_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

df = remove_outliers_iqr(df)

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

target_column = 'target'
X = df.drop(target_column, axis=1)
y = df[target_column]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

selector = SelectKBest(score_func=chi2, k=8)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = X.columns[selector.get_support()]
print("Selected features:\n", selected_features)

scores = selector.scores_[selector.get_support()]
plt.figure(figsize=(10, 6))
sns.barplot(x=scores, y=selected_features)
plt.xlabel('Chi2 Score')
plt.title('Feature Importance Based on Chi2 Test')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# --- Custom Linear Regression Implementation ---
class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            loss = np.mean((y_pred - y) ** 2)
            self.losses.append(loss)

    def predict(self, X, threshold=0.5):
        y_pred = np.dot(X, self.weights) + self.bias
        return (y_pred >= threshold).astype(int)

# ---  Linear Regression ---
lr_model = CustomLinearRegression(learning_rate=0.01, n_iterations=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(" Linear Regression Accuracy:", lr_accuracy)
print(" Linear Regression Classification Report:\n", classification_report(y_test, lr_pred))
ConfusionMatrixDisplay(confusion_matrix(y_test, lr_pred)).plot(cmap='Blues')
plt.title(" Linear Regression Confusion Matrix")
plt.show()

# --- Custom Linear Regression Definition ---
class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):
            # Calculate predictions
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Calculate loss (Mean Squared Error)
            loss = np.mean((y_pred - y) ** 2)
            self.losses.append(loss)

    def predict(self, X, threshold=0.5):
        # Make predictions and apply threshold
        y_pred = np.dot(X, self.weights) + self.bias
        return (y_pred >= threshold).astype(int)


# Create and train the model
lr_model = CustomLinearRegression(learning_rate=0.01, n_iterations=1000)
lr_model.fit(X_train, y_train)

# Make predictions
lr_pred = lr_model.predict(X_test)

# Calculate accuracy and classification report
lr_accuracy = accuracy_score(y_test, lr_pred)
print("Linear Regression Accuracy:", lr_accuracy)
print("Linear Regression Classification Report:\n", classification_report(y_test, lr_pred))

# --- SVM with Grid Search ---
svm = SVC(probability=True)
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.001],
    'kernel': ['rbf']
}
grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_svm.fit(X_train, y_train)
best_svm = grid_svm.best_estimator_
print("Best SVM params:", grid_svm.best_params_)

svm_pred = best_svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print("SVM Accuracy:", svm_accuracy)
print("SVM Classification Report:\n", classification_report(y_test, svm_pred))
ConfusionMatrixDisplay(confusion_matrix(y_test, svm_pred)).plot(cmap='Greens')
plt.title("SVM Confusion Matrix")
plt.show()

# --- Decision Tree (ID3 style) ---
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print("Decision Tree Accuracy:", dt_accuracy)
print("Decision Tree Classification Report:\n", classification_report(y_test, dt_pred))
ConfusionMatrixDisplay(confusion_matrix(y_test, dt_pred)).plot(cmap='Oranges')
plt.title("Decision Tree Confusion Matrix")
plt.show()

# --- KNN with Grid Search ---
knn = KNeighborsClassifier()
param_grid_knn = {'n_neighbors': [3, 5, 7, 9]}
grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
grid_knn.fit(X_train, y_train)
best_knn = grid_knn.best_estimator_
print("Best KNN params:", grid_knn.best_params_)

knn_pred = best_knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
print("KNN Accuracy:", knn_accuracy)
print("KNN Classification Report:\n", classification_report(y_test, knn_pred))
ConfusionMatrixDisplay(confusion_matrix(y_test, knn_pred)).plot(cmap='Purples')
plt.title("KNN Confusion Matrix")
plt.show()

# --- Accuracy Comparison ---
model_names = ['Custom Linear Regression', 'SVM', 'Decision Tree', 'KNN']
accuracies = [lr_accuracy, svm_accuracy, dt_accuracy, knn_accuracy]

# تحضير بيانات DataFrame مع عمود hue لتفادي التحذير
accuracy_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracies
})

plt.figure(figsize=(10, 6))
sns.barplot(data=accuracy_df, x='Model', y='Accuracy', palette='Set2', hue='Model', dodge=False, legend=False)
plt.ylim(0, 1)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy Score')
plt.xticks(rotation=15)
plt.show()



