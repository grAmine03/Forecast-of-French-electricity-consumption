# README - Forecast of French electricity consumption with Deep Learning

Authors : Mohamed Amine GRINI and Marine VIEILARD

## Project Description
This project aims to predict electricity consumption in France based on historical consumption data from 2017 to 2021, as well as temporal and meteorological data. Several deep learning models have been tested, including Multi-Layer Perceptrons (MLP) and Convolutional Neural Networks (CNN) with attention mechanisms.

## Installation
### Prerequisites
- Python 3.8+
- Required libraries:
  ```bash
  pip install numpy pandas torch torchvision scikit-learn matplotlib seaborn holidays
  ```

## Project Structure
- `train.py`: Contains the full code for data preprocessing, model training, and evaluation.
- `train.csv` / `test.csv`: Training and test datasets.
- `meteo.parquet`: Meteorological data.
- `predict.py`: Script to make predictions on 2022 data.

## Usage
1. **Data Preprocessing**: The data is cleaned, missing values are imputed using `KNNImputer`. New temporal features are created, including sine and cosine encoding for `dayofyear`, `dayofweek`, and `hour`. French holidays and weekends are also included.
2. **Model Training**:
   ```bash
   python train.py
   ```
   - Ensure that `Models.py` is in the same directory.
   - 6-layer MLP and 6-layer MLP with batch normalization.
   - CNN with attention.
   - Optimization using Adam.
   - Custom RMSE loss `Custom_loss`: Sum of RMSE with different weights assigned to France, regions, and summer/winter seasons.
3. **Making Predictions**:
   ```bash
   python predict.py
   ```

## Models Used
- **MLP**: `MLP_Model` - Multi-layer perceptron model with dropout and `Tanh` activation.
- **MLP with Batch Normalization**: `MLP_Batch_Model` - Improved version with batch normalization.
- **CNN**: `CNNModel` - Simple convolutional model with pooling.
- **CNN with Attention**: `CNN_Attention` - Added attention layer and positional encoding.

## Results
- Optimized sum of RMSE on the test dataset.
- Visualization of errors and model performance.