# Fourier Transformer Model Card

## Model Details
- **Date Created**: April 2024
- **Version**: 1.0
- **Type**: Time Series Forecasting
- **Model Owner**: Zheling Zhang
- **Contact Information**: [Email](zheling.zhang@vanderbilt.edu)

## Intended Use
- **Primary Uses**: This model is intended for financial time series forecasting, specifically designed to predict future stock prices based on historical data.
- **Intended Users**: Data scientists, financial analysts, and quantitative researchers focused on stock market predictions.
- **Out-of-Scope Applications**: The model is not intended for use in non-time series data applications or for predicting categories or outcomes unrelated to stock prices.

## Factors
- **Relevant Technical Attributes**: The model processes numerical time series data and is optimized for financial datasets. It utilizes Fourier Transformations to capture cyclical patterns in stock price movements.
- **Relevant Demographic Groups**: Not applicable as the model deals with financial data without any direct human demographic implications.

## Training Data
- **Data Description**: The model was trained using historical stock price data from Tesla, Apple, Nvidia, and the S&P 500 index.
- **Time Period**: April 2023 to April 2024
- **Source**: Stock price data obtained from publicly available financial databases.

## Evaluation Data
- **Data Description**: The evaluation was conducted using the same stocks as the training data (Tesla, Apple, Nvidia, and S&P 500 index) to ensure consistency in performance measurement.
- **Time Period**: April 2023 to April 2024
- **Source**: As with the training data, the evaluation data was sourced from publicly available financial databases.