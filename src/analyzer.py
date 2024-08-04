import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import stats
from scipy.stats import ttest_ind
from matplotlib import colormaps
import streamlit as st

jamming_1 = pd.read_csv('active-gaussian-jamming/5805mhz_jamming_10dbm_gaussiannoise_1.csv')
jamming_2 = pd.read_csv('active-gaussian-jamming/5805mhz_jamming_10dbm_gaussiannoise_2.csv')
jamming_3 = pd.read_csv('active-gaussian-jamming/5805mhz_jamming_10dbm_gaussiannoise_3.csv')
benign_1 = pd.read_csv('active-benign-background/5ghz_activescan_background_loc1_1.csv')
benign_2 = pd.read_csv('active-benign-background/5ghz_activescan_background_loc1_2.csv')
benign_3 = pd.read_csv('active-benign-background/5ghz_activescan_background_loc1_3.csv')

# Cleaning and visualizing the data
def visualize_examples():

    jamming_test = jamming_1.drop('freq1', axis=1)
    jamming_test2 = jamming_test.drop('base_pwr_db', axis=1)
    sns.lineplot(jamming_test2)
    plt.title('Jamming example at 5.8 GHz')
    plt.show()

    benign_test = benign_1.drop(['freq1', 'base_pwr_db'], axis=1)
    sns.lineplot(benign_test)
    plt.title('Benign jamming at 5.8 GHz')
    plt.show()

    return plt

visualize_examples()

# Logistic regrression model with visualizations
def prediction_model():
    
    benign = pd.concat([benign_1, benign_2, benign_3], ignore_index=True)
    jamming = pd.concat([jamming_1, jamming_2, jamming_3], ignore_index=True)
    
    # Set binary values to benign and jamming
    benign['label'] = 0
    jamming['label'] = 1

    # Combine the data
    combined_data = pd.concat([benign, jamming], ignore_index=True)

    X = combined_data.drop(['label'], axis=1)
    y = combined_data['label']

    # Training the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    combined_model = LogisticRegression(max_iter=2000, random_state=42)
    combined_model.fit(X_train, y_train)

    combined_y_pred = combined_model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, combined_y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Jamming'], yticklabels=['Benign', 'Jamming'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

prediction_model()




