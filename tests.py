import pandas as pd
import numpy as np
from ant_is import run_colony
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import os
import json

# dataframe = pd.read_csv("databases/ecoli.csv", header=None)
# last_row = len(dataframe.columns) - 1
# classes = dataframe[last_row]
# dataframe = dataframe.drop(columns=[0, last_row])
# num_instances = len(dataframe.index)
# initial_pheromone = 1
# Q = 1
# evaporation_rate = 0.1
# X = dataframe.to_numpy()[0:8]
# print(classes.unique())
# Y = classes.to_numpy()[0:8]
# print(Y.shape)
from typing import *

CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1-score", "support"]


def get_stratfied_cross_validation_scores(scores: List[Dict]):
    new_score = scores[0].copy()
    keys = new_score.keys()
    test_length = len(scores)
    for key in keys:
        if isinstance(new_score[key], dict):
            for sub_key in new_score[key].keys():
                value = sum([i[key][sub_key] for i in scores]) / test_length
                new_score[key][sub_key] = value
        else:
            value = sum([i[key] for i in scores]) / test_length
            new_score[key] = value

    return new_score


def get_average_results(results, result_key, num_iterations):
    avg_results = {}
    for metric_name in CLASSIFICATION_METRICS[1:]:
        avg_results[metric_name] = sum([results[i][result_key]["macro avg"][metric_name] for i in results.keys()]) / \
                                  num_iterations

    avg_results["accuracy"] = sum([results[i][result_key]["accuracy"] for i in results.keys()]) / num_iterations
    return avg_results

def create_test(classifier,
                num_iterations: int,
                num_folds: int,
                X: np.ndarray,
                y: np.ndarray,
                X_valid,
                y_valid,
                output_name: str,
                reduce_instances: bool = False,
                initial_pheromone: float = None,
                evaporation_rate: float = None,
                Q: float = None, ):
    num_instances = X.shape[0]
    results = {}
    for i in range(num_iterations):
        print(f"beginning of iterarion {i}")
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True)
        partial_scores = []
        valid_scores = []
        selected_indices = []
        for train_indices, test_indices in folds.split(X, y):
            train_set = train_indices
            if reduce_instances:
                reduced_indices = run_colony(X[train_indices], y[train_indices], initial_pheromone, evaporation_rate, Q)
                selected_indices.append(reduced_indices.tolist())
                train_set = reduced_indices

            classifier.fit(X[train_set], y[train_set])
            y_pred = classifier.predict(X[test_indices])
            y_pred_valid = classifier.predict(X_valid)
            score = classification_report(y[test_indices], y_pred, output_dict=True, zero_division=1)
            valid_score = classification_report(y_valid, y_pred_valid, output_dict=True, zero_division=1)
            valid_scores.append(valid_score)
            partial_scores.append(score)

        scores = get_stratfied_cross_validation_scores(partial_scores)
        v_score = get_stratfied_cross_validation_scores(valid_scores)
        print(f"accuracy: {scores['accuracy']} ---- {v_score['accuracy']}")
        for metric_name in scores["macro avg"].keys():
            print(f"{metric_name}: {scores['macro avg'][metric_name]} ---- {v_score['macro avg'][metric_name]}")

        print("-------------------------------------------------------")
        new_result = "i" + str(i)
        results[new_result] = {
            "scores": get_stratfied_cross_validation_scores(partial_scores),
            "valid_scores": get_stratfied_cross_validation_scores(valid_scores),
            "partial_scores": partial_scores,
            "partial_valid_scores": valid_scores,
        }

        if reduce_instances:
            results[new_result]["selected_indices"] = selected_indices

    avg_scores = get_average_results(results, "scores", num_iterations)
    avg_valid_scores = get_average_results(results, "valid_scores", num_iterations)

    results["num_iterations"] = num_iterations
    results["num_folds"] = num_folds
    results["strategy"] = "Strafied Cross Validation"  # For a while is the unique validation method availble in tests
    results["avg_scores"] = avg_scores
    results["avg_valid_scores"] = avg_valid_scores
    print("---==Final test results==---")
    for key in avg_scores.keys():
        print(f"{key}: {avg_scores[key]} ---- {avg_valid_scores[key]}")


    print("Generating output JSON file....")
    json.dump(results, open(output_name, 'w'), indent=2)


def nn_test(X, y, X_valid, y_valid, k, num_iterations, out_name, reduce_instance = False):
    classifier = KNeighborsClassifier(n_neighbors=k)
    create_test(classifier, num_iterations, 10, X, y, X_valid, y_valid, out_name, reduce_instance, 1, 0.1, 1)


def cart_test(X, y, X_valid, y_valid, num_iterations, out_name, reduce_instance = False):
    classifier = DecisionTreeClassifier()
    create_test(classifier, num_iterations, 10, X, y, X_valid, y_valid, out_name, reduce_instance, 1, 0.1, 1)


def svm_test(X, y, X_valid, y_valid, num_iterations, out_name, reduce_instance=False):
    classifier = SVC()
    create_test(classifier, num_iterations, 10, X, y, X_valid, y_valid, out_name, reduce_instance, 1, 0.1, 1)


def naive_bayes_test(X, y, X_valid, y_valid,num_iterations, out_name, reduce_instance = False):
    classifier = GaussianNB()
    create_test(classifier, num_iterations, 10, X, y, X_valid, y_valid, out_name, reduce_instance, 1, 0.1, 1)


def random_forest_test(X, y, X_valid, y_valid, num_iterations, out_name, reduce_instance = False):
    classifier = RandomForestClassifier()
    create_test(classifier, num_iterations, 10, X, y, X_valid, y_valid, out_name, reduce_instance, 1, 0.1, 1)

def mlp_test(X, y, X_valid, y_valid, num_iterations, out_name, reduce_instance = False):
    classifier = MLPClassifier()
    create_test(classifier, num_iterations, 10, X, y, X_valid, y_valid, out_name, reduce_instance, 1, 0.1, 1)

def run_all_tests(X, y, X_valid, y_valid, term_output):
    print("RUNNING TEST FOR " + term_output)
    print("1-NN Test")
    nn_test(X, y, X_valid, y_valid, 1, 1, f'outputs/full_{term_output}_1nn_results.json')
    nn_test(X, y, X_valid, y_valid, 1, 10, f'outputs/reduced_{term_output}_1nn_results.json', reduce_instance=True)
    print("Guassian Naive Bayes Test")
    naive_bayes_test(X, y, X_valid, y_valid, 1, f'outputs/full_{term_output}_nb_results.json')
    naive_bayes_test(X, y, X_valid, y_valid, 10, f'outputs/reduced_{term_output}_nb_results.json', reduce_instance=True)
    print("CART test")
    cart_test(X, y, X_valid, y_valid, 1, f'outputs/full_{term_output}_cart_results.json')
    cart_test(X, y, X_valid, y_valid, 10, f'outputs/reduced_{term_output}_cart_results.json', reduce_instance=True)
    print("SVM Test")
    svm_test(X, y, X_valid, y_valid, 1, f'outputs/full_{term_output}_svm_results.json')
    svm_test(X, y, X_valid, y_valid, 10, f'outputs/reduced_{term_output}_svm_results.json', reduce_instance=True)
    print("MLP Test")
    mlp_test(X, y, X_valid, y_valid, 1, f'outputs/full_{term_output}_mlp_results.json')
    mlp_test(X, y, X_valid, y_valid, 10, f'outputs/reduced_{term_output}_mlp_results.json', reduce_instance=True)
    print("Random Forest Test")
    random_forest_test(X, y, X_valid, y_valid, 1, f'outputs/full_{term_output}_random_forest_results.json')
    random_forest_test(X, y, X_valid, y_valid, 10, f'outputs/reduced_{term_output}_random_forest_results.json', reduce_instance=True)




def main():

    # Test esc
    df = pd.read_csv("databases/AG/Escrita/TreinamentoDesbalanceadoEscPreprocessada.csv", sep=";")
    df_valid = pd.read_csv("databases/AG/Escrita/TesteEscPreprocessada.csv", sep=";")
    y = df["TDE_MG_Esc"].to_numpy()
    X = df.drop(columns=["TDE_MG_Esc"]).to_numpy()
    y_valid = df_valid["TDE_MG_Esc"].to_numpy()
    X_valid = df_valid.drop(columns=["TDE_MG_Esc"]).to_numpy()
    run_all_tests(X, y, X_valid, y_valid, "esc")

    # Test Leit
    df = pd.read_csv("databases/AG/Leitura/TreinamentoDesbalanceadoLeitPreprocessada.csv", sep=";")
    df_valid = pd.read_csv("databases/AG/Leitura/TesteLeitPreprocessada.csv", sep=";")
    y = df["TDE_MG_Leit"].to_numpy()
    X = df.drop(columns=["TDE_MG_Leit"]).to_numpy()
    y_valid = df_valid["TDE_MG_Leit"].to_numpy()
    X_valid = df_valid.drop(columns=["TDE_MG_Leit"]).to_numpy()
    run_all_tests(X, y, X_valid, y_valid, "leit")












if __name__ == '__main__':
    main()
