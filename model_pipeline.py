import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def prepare_data(train_file, test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Séparation des features et de la cible
    target = 'Churn'  # Mets ici le nom exact de la colonne cible
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # Encodage des colonnes catégoriques
    label_encoders = {}
    for col in X_train.columns:
        if X_train[col].dtype == 'object':  # Vérifie si la colonne est catégorique
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
            label_encoders[col] = le  # Stocke l'encodeur pour réutilisation

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, max_depth=None):
    """
    Entraîne un modèle Decision Tree sur les données d'entraînement.
    """
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model


def save_model(model, filename="decision_tree_model.pkl"):
    """
    Sauvegarde le modèle entraîné dans un fichier.
    """
    joblib.dump(model, filename)
    print(f"Modèle sauvegardé sous {filename}")


def load_model(filename="decision_tree_model.pkl"):
    """
    Charge un modèle sauvegardé.
    """
    return joblib.load(filename)


def evaluate_model(model, X_test, y_test):
    """Évaluer le modèle et afficher les métriques de performance."""
    y_pred = model.predict(X_test)