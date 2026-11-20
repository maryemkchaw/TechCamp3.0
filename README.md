# Détection d’anomalies végétales avec MobileNetV2

Ce projet entraîne un classificateur binaire à partir du jeu de données PlantVillage. Il distingue les feuilles **normales** des feuilles **anormales** à l’aide du transfer learning avec MobileNetV2, puis exporte le modèle au format TensorFlow Lite pour un déploiement potentiel sur Raspberry Pi.

## Fonctionnalités

- Préparation des classes `NORMAL` et `ANOMALY` à partir des catégories PlantVillage (`healthy` devient `NORMAL`, les autres `ANOMALY`).
- Séparation stratifiée par classe en ensembles d’entraînement (70 %), de validation (15 %) et de test (15 %).
- Entraînement MobileNetV2 avec augmentation d’images et pondération des classes.
- Sélection du seuil de classification sur l’ensemble de validation uniquement, puis évaluation finale sur le test.
- Export du modèle TensorFlow Lite, des étiquettes, de la configuration et des métriques.

## Contenu

- [`TechCamp_CIS_Plant_Anomaly_MobileNetV2_FIXED-1.ipynb`](TechCamp_CIS_Plant_Anomaly_MobileNetV2_FIXED-1.ipynb) — notebook complet de préparation, entraînement, évaluation et export.

## Exécution dans Google Colab

1. Ouvrir le notebook dans Google Colab et sélectionner un environnement avec GPU si disponible.
2. Exécuter les cellules dans l’ordre. Le notebook installe `scikit-learn` et demande l’accès à Google Drive.
3. Le jeu de données est cloné dans le stockage temporaire de Colab depuis le [miroir PlantVillage utilisé par le notebook](https://github.com/gabrieldgf4/PlantVillage-Dataset). Il n’est pas copié dans ce dépôt ni dans Drive.
4. Les modèles et résultats sont enregistrés dans `MyDrive/TECHCAMP_CIS/`, sous `models/` et `results/`.

Par défaut, le notebook limite à 300 le nombre d’images par classe d’origine pour réduire le temps d’exécution. Ce réglage peut être modifié dans le notebook pour utiliser toutes les images.

## Fichiers exportés

Le dossier `models/` de Google Drive contient notamment :

- `best_mobilenetv2_fixed.keras` — meilleur modèle Keras sauvegardé pendant l’entraînement.
- `plant_anomaly_mobilenetv2_fixed.tflite` — modèle TensorFlow Lite.
- `labels.txt` et `config.json` — classes, taille d’entrée, prétraitement et seuil retenu.

Le notebook enregistre également les métriques et l’historique d’entraînement dans `results/`.

## Prédiction et interprétation

L’ordre des classes est fixe : `0 = NORMAL`, `1 = ANOMALY`. La sortie sigmoïde représente la probabilité d’anomalie. Pour l’inférence, utiliser le seuil écrit dans `config.json` : une probabilité supérieure ou égale au seuil indique `ANOMALY`.

## Limites

Il s’agit d’un prototype d’apprentissage. Les performances sur PlantVillage ne garantissent pas les mêmes résultats sur des images prises en conditions réelles ; une validation sur des données de terrain est nécessaire avant tout usage opérationnel. Les conditions d’utilisation du jeu de données tiers s’appliquent séparément.
