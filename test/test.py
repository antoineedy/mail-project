import sys
import os
import json

# setting path
sys.path.append("../mail-project")

from pipeline.first_classifier import first_classifier

from tqdm import tqdm

# First classifier test

from sklearn.metrics import confusion_matrix, classification_report

# 1 - Disponibilité des chambres


def test_router(datasource, queries):
    nb = {
        "disponibilite_chambres": 0,
        "ouverture_accueil": 0,
        "activite": 0,
        "other": 0,
        "error": 0,
    }
    for query in tqdm(queries):
        query = queries[query]
        found = first_classifier.invoke(query)["datasource"]
        nb[found] += 1
        if found != datasource:
            print(f"Query : {query}")
            print(f"Expected : {datasource}")
            print(f"Found : {found}")
            print("-----------------")

    print(nb)


to_test = [True, True, True]

if to_test[0]:
    with open("test/test_data/queries_ouverture_accueil.json", "r") as f:
        queries = json.load(f)
    test_router("ouverture_accueil", queries)
if to_test[1]:
    with open("test/test_data/queries_disponibilite_chambres.json", "r") as f:
        queries = json.load(f)
    test_router("disponibilite_chambres", queries)
if to_test[2]:
    with open("test/test_data/queries_activite.json", "r") as f:
        queries = json.load(f)
    test_router("activite", queries)
