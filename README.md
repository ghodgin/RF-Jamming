# Predicting and classifying malicious RF signals

![rf signals gif](https://www.compasseur.com/wp-content/uploads/2023/06/3_livecheck.gif)

## Table of contents
1. [Overview](#overview)
2. [Hypothesis](#hypothesis)
3. [The Data](#the-data)
4. [Visualizations](#visualizations)
5. [Conclusion](#conclusion)

## Overview
The RF spectrum plays a pivotal role in modern communications, and with the increasing prevalence of malicious RF jamming attacks, the reliability and security of wireless systems are under significant threat. To safeguard your critical frequencies, it is essential to accurately detect and identify any disruptions, whether caused by deliberate interference or other factors. This requires robust monitoring and analysis to ensure that your communications remain secure and uninterrupted.

## Hypothesis
## Null Hypothesis
- Active scan jamming techniques are as difficult to identify and classify as passive scan techniques.

## Alternate Hypothesis
- Active scan jamming techniques are easier to identify and classify than passive scan techniques.

## The Data
Data Overview: 
- The data was presented in a 14.5 GB zip file, with 96k spectral scans. (5k active scans, 91k passive scans)
- Format: .csv
- Captures instances of RF jamming (malicious) and non-jamming (benign) scenarios using a Raspberry Pi4 device 
- 'Jamming' and 'floor' data were collected in an RF chamber to ensure controlled conditions
- 'Background' data was obtained from real-world environments, with varying interference scenarios to simulate typical usage scenarios
- For the sake of this project, three .csv files were selected from the different scenarios listed above. (Active benign background, active gaussian jamming, passive benign background, and passive gaussian jamming)

Data Features: 
- freq1: Frequency bin 1
- noise: Noise level
- max_magnitude: Maximum magnitude of the signal
- total_gain_db: Total gain in dB
- base_pwr_db: Base power in dB
- rssi: Received signal strength indicator
- relpwr_db: Relative power in dB
- avgpwr_db: Average power in dB
    


## Visualizations

## Conclusion

After analyzing the metrics from the prediction model, I would reject the null hypothesis. It is easier to detect Active jamming in real world conditions, compared to Passive jamming techniques. This means that if someone wanted to maliciously 'attack' a frequency, they would be better off focusing their efforts into passive methods. This also suggests that people who are looking for disturbances on their own frequencies, that a good first step would be to look out for any passive scans on the same frequency. 