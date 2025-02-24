# Stock Prediction Algorithm - Project No. 23-2-1-2973

## Project Description

The goal of this project is to develop a machine learning-based algorithm for predicting stock prices. The algorithm integrates various techniques from deep learning and time series forecasting, with a focus on the LSTM model. It is designed to be flexible and provide forecasts for any stock with available historical data.

## Key Features
- Use of an API to collect historical stock data.
- Support for multi-step forecasting.
- Performance evaluation using error metrics such as MAE, RMSE, and MAPE.
- User-friendly GUI for inputting stock data and viewing graphical forecasts.

## Project Structure
- **Data Collection**: Fetching data from the Financial Modeling Prep API.
- **Data Processing**: Data filtering, handling missing values, and applying resampling.
- **Model Training & Evaluation**: Training models and comparing performance.
- **Prediction & Visualization**: Generating forecasts and presenting results graphically.

## System Requirements
- Python 3.x
- Required Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `tensorflow`
  - `sklearn`
  - `requests`

## Results & Conclusions
- The algorithm achieved an average prediction accuracy of 86%.
- Accuracy decreases as the forecast horizon increases.
- Further improvements can be made by optimizing parameters and integrating additional models.

## Future Development Suggestions
- Predicting entire market sectors instead of individual stocks.
- Integrating algorithms for technical pattern recognition in trading.
- Developing a version that supports options and cryptocurrencies.

## Additional Documentation
The project is documented on GitHub: [ProphitProphet](https://github.com/sivanf8/ProphitProphet)

Authors: Sivan Farada & Shlomi Kuzari
