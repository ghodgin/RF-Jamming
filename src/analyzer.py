import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
import streamlit as st

# Prediction model for maliciously jammed signals, as well as a deployed streamlit application
# Capstone project

# Streamlit application
def streamlit_app():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Home", "Visualize Data"])

    if page == "Home":
        st.title("RF Jamming Predictions")
        st.write("By Greg Hodgin")
        st.markdown('![rf signals gif](https://www.compasseur.com/wp-content/uploads/2023/06/3_livecheck.gif)')
        st.write("Welcome to the RF Jamming visualization App. Use the sidebar to navigate to the visualization page.")
        st.write('## Overview')
        st.write('The RF spectrum plays a pivotal role in modern communications, and with the increasing prevalence of malicious RF jamming attacks, the reliability and security of wireless systems are under significant threat. To safeguard your critical frequencies, it is essential to accurately detect and identify any disruptions, whether caused by deliberate interference or other factors. This requires robust monitoring and analysis to ensure that your communications remain secure and uninterrupted.')
        st.write('## Hypothesis')
        st.write('### Null Hypothesis')
        st.write('- Received signal strength is NOT a strong indicator of jamming')
        st.write('### Alternate Hypothesis')
        st.write('- Received signal strength is a strong indicator of jamming in both active and passive jamming scenarios')
        st.write('## The Data')
        st.write('- The data was presented in a 14.5 GB zip file, with 96k spectral scans. (5k active scans, 91k passive scans)')
        st.write('- Format: .csv')
        st.write('- Captures instances of RF jamming (malicious) and non-jamming (benign) scenarios using a Raspberry Pi4 device ')
        st.write('- Jamming and floor data were collected in an RF chamber to ensure controlled conditions')
        st.write('- Background data was obtained from real-world environments, with varying interference scenarios to simulate typical usage scenarios')
        st.write('- For the sake of this project, three .csv files were selected from the different scenarios listed above. (Active benign background, active gaussian jamming, passive benign background, and passive gaussian jamming)')
        st.write('Data Features:')
        st.write('- freq1: Frency bin 1')
        st.write('- noise: Noise level')
        st.write('- total_gain_db: Total gain in dB')
        st.write('- base_pwr_db: Base power in dB')
        st.write('- rssi: Received signal strength indicator')
        st.write('- relpwr_dB: Relative power in dB')
        st.write('- avgpwr_dB: Average power in dB')
        st.write('## Visualizations')
        st.image('images/Figure_1.png')
        st.image('images/Figure_2.png')
        st.image('images/Figure_3.png')
        st.image('images/Figure_4.png')
        st.write('## Metrics')
        st.write('The output from the prediction model and hypothesis tests are as follows:')
        st.write('Active Metrics:')
        st.write('- Accuracy: 0.96')
        st.write('- Precision: 0.98')
        st.write('- Recall: 0.96')
        st.write('- F1 Score: 0.97')
        st.write('Passive Metrics:')
        st.write('- Accuracy: 0.90')
        st.write('- Precision: 0.90')
        st.write('- Recall: 0.91')
        st.write('- F1 Score: 0.91')
        st.write('Hypothesis tests on RSSI')
        st.write('- Active RSSI T-Statistic: -373.330')
        st.write('- Active RSSI P-Value: 0.0')
        st.write('- Passive RSSI T-Statistic: -57.163')
        st.write('- Passive RSSI P-Value: 0.0')
        st.write('## Conclusion')
        st.write('After analyzing the metrics and visualizations, it is clear that the logisitic regression model used is very accurate in predicting if signals from this dataset are being maliciously jammed or not. Additionally, the model also shows that it can predict the active gaussian jamming easier than the passive, though not by a large margin. (The active jamming accuracy was 96%, whereas the passive jamming accuracy was 90%.) This leads into the hypothesis test, where I tested to see if the RSSI (Received signal strength indicator) was a good indicator of whether or not a signal is being maliciously jammed. Both tests show that there is a clear relationship in increased RSSI being a tell-tell sign of a signal being maliciously jammed, which is also visualized in the examples above. With all of this being said, we can reject the null hypothesis of RSSI not being a good indicator of malicious jamming.')


    elif page == "Visualize Data":
        st.title("Visualize Data")

        # List of csv selections
        file_options = {
            "Signal 1" : "data/active-gaussian-jamming/5805mhz_jamming_10dbm_gaussiannoise_1.csv",
            "Signal 2" : "data/active-gaussian-jamming/5805mhz_jamming_neg10dbm_gaussiannoise_9.csv",
            "Signal 3" : "data/active-gaussian-jamming/5805mhz_jamming_0dbm_gaussiannoise_28.csv",
            "Signal 4" : "data/active-benign-background/5ghz_activescan_background_loc1_1.csv",
            "Signal 5" : "data/active-benign-background/5ghz_activescan_background_loc2_35.csv",
            "Signal 6" : "data/active-benign-background/5ghz_activescan_floor_32.csv",
            "Signal 7" : "data/passive-benign-background/2.4ghz_passivescan_background_loc1_2.csv",
            "Signal 8" : "data/passive-benign-background/2.4ghz_passivescan_background_loc1_18.csv",
            "Signal 9" : "data/passive-benign-background/2.4ghz_passivescan_background_loc1_27.csv",
            "Signal 10" : "data/passive-gaussian-jamming/2412mhz_jamming_6dbm_gaussiannoise_13.csv",
            "Signal 11" : "data/passive-gaussian-jamming/2412mhz_jamming_9dbm_gaussiannoise_6.csv",
            "Signal 12" : "data/passive-gaussian-jamming/2412mhz_jamming_12dbm_gaussiannoise_1.csv"
        }

        correct_options = {
            "Signal 1" : True,
            "Signal 2" : True,
            "Signal 3" : True,
            "Signal 4" : False,
            "Signal 5": False,
            "Signal 6" : False,
            "Signal 7" : False, 
            "Signal 8" : False,
            "Signal 9" : False,
            "Signal 10" : True,
            "Signal 11" : True,
            "Signal 12" : True
        }

        # User selects a CSV to look at
        selected_name = st.selectbox("Select a signal", list(file_options))
        selected_file = file_options[selected_name]
        
        if selected_file:
            # Loads and cleans
            data = pd.read_csv(selected_file)
            cleaned_data = data.drop(['freq1', 'base_pwr_db', 'noise'], axis=1)
            
            # Displays raw data
            st.write("Raw Data")
            st.write(data.head())

            # Visualizes the signal
            st.write("Signal Visualization")
            st.line_chart(cleaned_data)

            st.write("Is this signal being maliciously jammed?")
            st.write("HINT: remember RSSI!")

            # Adding two columns for yes / no buttons
            col1, col2 = st.columns(2)
            
            # Yes / no buttons for visualization game
            with col1:
                if st.button("Yes"):
                    if correct_options[selected_name]:
                        st.write("Correct!")
                    else:
                        st.write("Incorrect")
            with col2:
                if st.button("No"):
                    if not correct_options[selected_name]:
                        st.write("Correct!")
                    else: st.write("Incorrect")

# Active jamming data
jamming_1 = pd.read_csv('data/active-gaussian-jamming/5805mhz_jamming_10dbm_gaussiannoise_1.csv')
jamming_2 = pd.read_csv('data/active-gaussian-jamming/5805mhz_jamming_0dbm_gaussiannoise_28.csv')
jamming_3 = pd.read_csv('data/active-gaussian-jamming/5805mhz_jamming_neg10dbm_gaussiannoise_9.csv')

benign_1 = pd.read_csv('data/active-benign-background/5ghz_activescan_background_loc1_1.csv')
benign_2 = pd.read_csv('data/active-benign-background/5ghz_activescan_background_loc2_35.csv')
benign_3 = pd.read_csv('data/active-benign-background/5ghz_activescan_floor_32.csv')

# Passive jamming data
passive_benign1 = pd.read_csv('data/passive-benign-background/2.4ghz_passivescan_background_loc1_2.csv')
passive_benign2 = pd.read_csv('data/passive-benign-background/2.4ghz_passivescan_background_loc1_18.csv')
passive_benign_3 = pd.read_csv('data/passive-benign-background/2.4ghz_passivescan_background_loc1_27.csv')

passive_jamming1 = pd.read_csv('data/passive-gaussian-jamming/2412mhz_jamming_12dbm_gaussiannoise_1.csv')
passive_jamming2 = pd.read_csv('data/passive-gaussian-jamming/2412mhz_jamming_9dbm_gaussiannoise_6.csv')
passive_jamming3 = pd.read_csv('data/passive-gaussian-jamming/2412mhz_jamming_6dbm_gaussiannoise_13.csv')

# Cleaning and visualizing the data
def visualize_examples():

    jamming_test = jamming_1.drop('freq1', axis=1)
    jamming_test2 = jamming_test.drop('base_pwr_db', axis=1)
    sns.lineplot(jamming_test2)
    plt.title('Malicious Jamming Example at 5.8 GHz')
    # plt.show() (remove # if needed)

    benign_test = benign_1.drop(['freq1', 'base_pwr_db'], axis=1)
    sns.lineplot(benign_test)
    plt.title('Benign jamming at 5.8 GHz')
    # plt.show()
    
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
    

    accuracy_passive = accuracy_score(y_test_passive, passive_y_pred)
    precision_passive = precision_score(y_test_passive, passive_y_pred)
    recall_passive = recall_score(y_test_passive, passive_y_pred)
    f1_passive = f1_score(y_test_passive, passive_y_pred)

    return accuracy_passive, precision_passive, recall_passive, f1_passive

# Hypothesis tests
def hypothesis_test_active():
    active_benign_data = pd.concat([benign_1, benign_2, benign_3], ignore_index=True)
    active_jamming_data = pd.concat([jamming_1, jamming_2, jamming_3], ignore_index=True)

    active_benign_rssi = active_benign_data['rssi']
    active_jamming_rssi = active_jamming_data['rssi']

    t_stat, p_value = stats.ttest_ind(active_benign_rssi, active_jamming_rssi)

    print(f'Active RSSI T-Statistic: {t_stat}')
    print(f'Active RSSI P-Value: {p_value}')

def hypothesis_test_passive():
    passive_benign_data = pd.concat([passive_benign1, passive_benign2, passive_benign_3], ignore_index=True)
    passive_jamming_data = pd.concat([passive_jamming1, passive_jamming2, passive_jamming3], ignore_index=True)

    passive_benign_rssi = passive_benign_data['rssi']
    passive_jamming_rssi = passive_jamming_data['rssi']

    t_stat, p_value = stats.ttest_ind(passive_benign_rssi, passive_jamming_rssi)

    print(f'Passive RSSI T-Statistic: {t_stat}')
    print(f'Passive RSSI T-Statistic: {p_value}')


active_metrics = active_prediction_model()
passive_metrics = passive_prediction_model()

# Print metrics from above models
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
    visualize_examples()
    active_prediction_model()
    passive_prediction_model()
    hypothesis_test_active()
    hypothesis_test_passive()