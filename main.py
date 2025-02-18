from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model
import argparse

def main():
     # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Decision Tree Churn Prediction")
    parser.add_argument("--train_file", type=str, default="churn-bigml-80.csv", help="Train file path")
    parser.add_argument("--test_file", type=str, default="churn-bigml-20.csv", help="Test file path")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the tree")
    parser.add_argument("--save", action='store_true', help="Save the trained model")
    args = parser.parse_args()

    # Préparation des données
    X_train, X_test, y_train, y_test = prepare_data(args.train_file, args.test_file)

    # Entraînement du modèle
    model = train_model(X_train, y_train, max_depth=args.max_depth)

    # Évaluation du modèle
    accuracy = evaluate_model(model, X_test, y_test)

    # Affichage de la précision
    print(f"Précision du modèle : {accuracy:.2f}")

    # Sauvegarde du modèle si demandé
    if args.save:
        save_model(model)

if __name__ == "__main__":
    main()







