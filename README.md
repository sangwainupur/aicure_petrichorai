# Heart Rate Prediction Model

This repository contains an advanced model for predicting an individual's heart rate based on diverse attributes derived from ECG recordings. The dataset used for training is provided in train_data.csv.

## Files and Directories

- *train_data.csv:* Dataset used for training the model.
- *run.py:* Script to predict heart rates for new data.
- *sample_test_data.csv:* Sample test data for reference.
- *sample_output_generated.csv:* Expected output for the sample test data.
- *check_new_fin.pkl:* the final model (check_new_fin.pkl) and its parameters.
- *aicure_petrichorai.ipynb:* the final jupyter notebook used to train the model check_new_fin.pkl and contains details on data exploration, feature engineering, and model training.

## Model Development

- The heart rate prediction model is constructed using diverse features extracted from ECG recordings.
- Refer to the Jupyter notebook (aicure_petrichorai.ipynb) for details on data exploration, feature engineering, and model training.

## Running the Model

To predict heart rates for new data, use the provided run.py script. Follow the instructions below:

### Prerequisites

- Python 3.x
- Required Python libraries (specified in requirements.txt)

### Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/heart-rate-prediction.git
   cd heart-rate-prediction
2. Run the run.py and generate the results.csv in the terminal:
   ```bash
   python run.py test_data.csv
