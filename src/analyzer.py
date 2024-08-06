import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import stats
from scipy.stats import ttest_ind
from matplotlib import colormaps
import streamlit as st

def streamlit_app():
    st.title('RF Spectrum Analyzer')
    st.write("Hello streamlit")

# Active jamming data
jamming_1 = pd.read_csv('active-gaussian-jamming/5805mhz_jamming_10dbm_gaussiannoise_1.csv')
jamming_2 = pd.read_csv('active-gaussian-jamming/5805mhz_jamming_0dbm_gaussiannoise_28.csv')
jamming_3 = pd.read_csv('active-gaussian-jamming/5805mhz_jamming_neg10dbm_gaussiannoise_9.csv')

benign_1 = pd.read_csv('active-benign-background/5ghz_activescan_background_loc1_1.csv')
benign_2 = pd.read_csv('active-benign-background/5ghz_activescan_background_loc2_35.csv')
benign_3 = pd.read_csv('active-benign-background/5ghz_activescan_floor_32.csv')

# Passive jamming data
passive_benign1 = pd.read_csv('passive-benign-background/2.4ghz_passivescan_background_loc1_2.csv')
passive_benign2 = pd.read_csv('passive-benign-background/2.4ghz_passivescan_background_loc1_18.csv')
passive_benign_3 = pd.read_csv('passive-benign-background/2.4ghz_passivescan_background_loc1_27.csv')

passive_jamming1 = pd.read_csv('passive-gaussian-jamming/2412mhz_jamming_12dbm_gaussiannoise_1.csv')
passive_jamming2 = pd.read_csv('passive-gaussian-jamming/2412mhz_jamming_9dbm_gaussiannoise_6.csv')
passive_jamming3 = pd.read_csv('passive-gaussian-jamming/2412mhz_jamming_6dbm_gaussiannoise_13.csv')

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

# Logistic regression model with visualizations
def active_prediction_model():
    
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
    plt.title('Confusion Matrix of Active Scans')
    plt.show()

    accuracy = accuracy_score(y_test, combined_y_pred)
    precision = precision_score(y_test, combined_y_pred)
    recall = recall_score(y_test, combined_y_pred)
    f1 = f1_score(y_test, combined_y_pred)

    return accuracy, precision, recall, f1


# Same model as above, just for passive scans
def passive_prediction_model():


    passive_benign = pd.concat([passive_benign1, passive_benign2, passive_benign_3], ignore_index=True)
    passive_jamming = pd.concat([passive_jamming1, passive_jamming2, passive_jamming3], ignore_index=True)

    # Set binary values to benign and jamming
    passive_benign['label'] = 0
    passive_jamming['label'] = 1

    # Combine the data
    combined_passive = pd.concat([passive_benign, passive_jamming])

    passive_X = combined_passive.drop(['label'], axis=1)
    passive_y = combined_passive['label']

    # Training the model
    X_train_passive, X_test_passive, y_train_passive, y_test_passive = train_test_split(passive_X, passive_y, test_size=0.3, random_state=42)

    passive_model = LogisticRegression(max_iter=2000, random_state=42)
    passive_model.fit(X_train_passive, y_train_passive)

    passive_y_pred = passive_model.predict(X_test_passive)

    # Confusion matrix
    cm = confusion_matrix(y_test_passive, passive_y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Jamming'], yticklabels=['Benign', 'Jamming'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix of Passive Scans')
    plt.show()

    accuracy_passive = accuracy_score(y_test_passive, passive_y_pred)
    precision_passive = precision_score(y_test_passive, passive_y_pred)
    recall_passive = recall_score(y_test_passive, passive_y_pred)
    f1_passive = f1_score(y_test_passive, passive_y_pred)

    return accuracy_passive, precision_passive, recall_passive, f1_passive


active_metrics = active_prediction_model()
passive_metrics = passive_prediction_model()

# Print metrics
print("Active Metrics:")
print(f"Accuracy: {active_metrics[0]: }")
print(f"Precision: {active_metrics[1]: }")
print(f"Recall: {active_metrics[2]: }")
print(f"F1 Score: {active_metrics[3]: }")

print("\nPassive Metrics:")
print(f"Accuracy: {passive_metrics[0]: }")
print(f"Precision: {passive_metrics[1]: }")
print(f"Recall: {passive_metrics[2]: }")
print(f"F1 Score: {passive_metrics[3]: }")


if __name__ == "__main__":
    streamlit_app()