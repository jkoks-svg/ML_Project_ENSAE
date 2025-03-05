# this file contains all package and functions that we create

# I importation des packages liées à l'analyse et la visualisation de données 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo 
  


# II importation des packages liées à la modélisation

import doubleml as dml
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier, XGBRegressor



# II-4. Visualisation des données manquantes


def missing_plot(df):
    """
    Visualise les valeurs manquantes avec un diagramme à barres et un heatmap.
    Utilise les bibliothèques seaborn.
    """
    # Heatmap des valeurs manquantes
    sns.heatmap(df.isna(), cbar=False)
    plt.title(r"$\bf{Heatmap\ of\ missing\ values}$", color="green")
    plt.show()