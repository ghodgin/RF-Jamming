import streamlit as st
import pandas as pd

# Visualizes signal
def visualize_signal(data):
    st.line_chart(data)

def main():
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

        st.write('## Conclusion')
        st.write('After analyzing the metrics from the prediction model, I would reject the null hypothesis. It is easier to detect Active jamming in real world conditions, compared to Passive jamming techniques. This means that if someone wanted to maliciously attack a frequency, they would be better off focusing their efforts into passive methods. This also suggests that people who are looking for disturbances on their own frequencies, that a good first step would be to look out for any passive scans on the same frequency.')


    elif page == "Visualize Data":
        st.title("Visualize Data")

        # List of csv selections
        csv_files = [
            "data/active-gaussian-jamming/5805mhz_jamming_10dbm_gaussiannoise_1.csv",
            "data/active-gaussian-jamming/5805mhz_jamming_neg10dbm_gaussiannoise_9.csv",
            "data/active-gaussian-jamming/5805mhz_jamming_0dbm_gaussiannoise_28.csv",
            "data/active-benign-background/5ghz_activescan_background_loc1_1.csv",
            "data/active-benign-background/5ghz_activescan_background_loc2_35.csv",
            "data/active-benign-background/5ghz_activescan_floor_32.csv",
            "data/passive-benign-background/2.4ghz_passivescan_background_loc1_2.csv",
            "data/passive-benign-background/2.4ghz_passivescan_background_loc1_18.csv",
            "data/passive-benign-background/2.4ghz_passivescan_background_loc1_27.csv",
            "data/passive-gaussian-jamming/2412mhz_jamming_6dbm_gaussiannoise_13.csv",
            "data/passive-gaussian-jamming/2412mhz_jamming_9dbm_gaussiannoise_6.csv",
            "data/passive-gaussian-jamming/2412mhz_jamming_12dbm_gaussiannoise_1.csv"
        ]

        # User selects a CSV to look at
        selected_file = st.selectbox("Select a CSV file", csv_files)
        
        if selected_file:
            # Loads and cleans
            data = pd.read_csv(selected_file)
            cleaned_data = data.drop(['freq1', 'base_pwr_db', 'noise'], axis=1)
            
            # Displays raw data
            st.write("Raw Data")
            st.write(data.head())

            # Visualizes the signal
            st.write("Signal Visualization")
            visualize_signal(cleaned_data)

if __name__ == "__main__":
    main()
