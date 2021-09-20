import pandas as pd
import numpy as np
from ant_is import run_colony
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

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


def create_test(classifier,
                num_iterations: int,
                num_folds: int,
                X: np.ndarray,
                y: np.ndarray,
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
        selected_indices = []
        for train_indices, test_indices in folds.split(X, y):
            train_set = train_indices
            if reduce_instances:
                reduced_indices = run_colony(X[train_indices], y[train_indices], initial_pheromone, evaporation_rate, Q)
                selected_indices.append(train_indices.tolist())
                train_set = reduced_indices

            classifier.fit(X[train_set], y[train_set])
            y_pred = classifier.predict(X[test_indices])
            score = classification_report(y[test_indices], y_pred, output_dict=True, zero_division=1)
            partial_scores.append(score)

        scores = get_stratfied_cross_validation_scores(partial_scores)
        print(f"accuracy: {scores['accuracy']}")
        for metric_name in scores["macro avg"].keys():
            print(f"{metric_name}: {scores['macro avg'][metric_name]}")

        print("-------------------------------------------------------")
        new_result = "i" + str(i)
        results[new_result] = {
            "scores": get_stratfied_cross_validation_scores(partial_scores),
            "partial_scores": partial_scores,
        }

        if reduce_instances:
            results[new_result]["selected_indices"] = selected_indices

    avg_scores = {}
    for metric_name in CLASSIFICATION_METRICS[1:]:
        avg_scores[metric_name] = sum([results[i]["scores"]["macro avg"][metric_name] for i in results.keys()]) / \
                                  num_iterations

    avg_scores["accuracy"] = sum([results[i]["scores"]["accuracy"] for i in results.keys()]) / num_iterations

    results["num_iterations"] = num_iterations
    results["num_folds"] = num_folds
    results["strategy"] = "Strafied Cross Validation"  # For a while is the unique validation method availble in tests
    results["avg_scores"] = avg_scores
    print("---==Final test results==---")
    for item in avg_scores.items():
        print(f"{item[0]}: {item[1]}")

    print("Generating output JSON file....")
    json.dump(results, open(output_name, 'w'), indent=2)


def nn_test(X, y, k, out_name, reduce_instance = False):
    classifier = KNeighborsClassifier(n_neighbors=k)
    create_test(classifier, 10, 3, X, y, out_name, reduce_instance, 1, 0.1, 1)


def cart_test(X, y, k, out_name, reduce_instance = False):
    classifier = DecisionTreeClassifier()
    create_test(classifier, 10, 3, X, y, out_name, reduce_instance, 1, 0.1, 1)


def svm_test(X, y, k, out_name, reduce_instance=False):
    classifier = SVC()
    create_test(classifier, 10, 3, X, y, out_name, reduce_instance, 1, 0.1, 1)



def naive_bayes_test(X, y, k, out_name, reduce_instance = False):
    classifier = GaussianNB()
    create_test(classifier, 10, 3, X, y, out_name, reduce_instance, 1, 0.1, 1)


def main():
    dataframe = pd.read_csv("databases/ecoli.csv", header=None)
    last_row = len(dataframe.columns) - 1
    classes = dataframe[last_row]
    dataframe = dataframe.drop(columns=[0, last_row])
    print("1-NN Test")
    nn_test(dataframe.to_numpy(), classes.to_numpy(), 1, 'outputs/full_ecoli_1nn_results.json')
    print("Guassian Naive Bayes Test")
    naive_bayes_test(dataframe.to_numpy(), classes.to_numpy(), 1, 'outputs/full_ecoli_nb_results.json')
    print("CART test")
    cart_test(dataframe.to_numpy(), classes.to_numpy(), 1, 'outputs/full_ecoli_cart_results.json')
    print("SVM Test")
    svm_test(dataframe.to_numpy(), classes.to_numpy(), 1, 'outputs/full_ecoli_svm_results.json')



if __name__ == '__main__':
    main()
