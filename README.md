# Fourier Transformer

## Introduction
This project aims to harness the power of the Fourier transformer, a variant of the linear transformer architecture, to predict stock prices. 
Utilizing advanced techniques in machine learning and signal processing, the Fourier transformer model in this project tackles the challenges of financial time-series forecasting. 
By transforming time-series data into the frequency domain, the model captures underlying patterns and trends that are essential for accurate predictions.

## Motivation
Financial markets are known for their volatility and unpredictability. 
Investors and analysts strive to predict market movements to make informed decisions, but traditional models often fall short in handling the complexity and noise inherent in stock price data. 
The Fourier transformer offers a promising solution by simplifying the computational complexity of traditional transformers while retaining their predictive power. 
This approach is particularly suited to processing long sequences of historical data, which is typical in stock market analysis.

The motivation for developing this project is twofold:
1. **Enhanced Prediction Accuracy:** By leveraging the Fourier transformer, this model aims to improve the accuracy of stock price forecasts, benefiting from the transformer's ability to manage long data sequences without the computational overhead of standard attention mechanisms.
2. **Real-time Processing Capability:** With increased efficiency, the model can process and predict stock prices in near real-time, a critical requirement for trading algorithms and investment strategies that rely on timely data.

## Model Structure Overview
![image](https://github.com/Zoliverling/Linear_Transformation/assets/106001844/d388e504-b32d-4d31-80f1-8a5f274d6f28)

### Traditional Transformer
The traditional transformer architecture consists of an encoder and decoder structure, each comprising multiple layers that perform complex functions. At the heart of the encoder and decoder lies the multi-head attention mechanism, which allows the model to focus on different parts of the input sequence when predicting each part of the output sequence. Each layer follows a specific order: multi-head attention is followed by a normalization step and a feed-forward network, with another normalization step thereafter. This architecture is powerful but computationally intensive, especially for long sequences, due to the quadratic complexity of the self-attention mechanism.

### Fourier Transformer
The Fourier transformer is an innovative adaptation of the traditional transformer that replaces the multi-head attention mechanism with a Fourier transform-based approach. The key distinction lies in how the attention is computed:
- Instead of calculating the attention weights for each element in the sequence with every other element (which is computationally expensive), the Fourier transform efficiently captures frequency-based relationships in the sequence.
- This not only reduces the computational complexity but also allows the model to handle much longer sequences without a significant increase in computation time.
- The feed-forward and normalization layers after the Fourier transform in the encoder remain unchanged, preserving the structure that enables complex representations.

### Comparison
- **Computational Efficiency:** The Fourier transformer is computationally more efficient than the traditional transformer. By converting the attention mechanism into the frequency domain, it sidesteps the need for quadratic computation with respect to sequence length.
- **Handling Long Sequences:** The Fourier transform's ability to work in the frequency domain means it can handle long-range dependencies and large sequences better than the traditional multi-head attention mechanism.
- **Preserved Depth and Complexity:** Despite the simplification in attention calculation, the depth and complexity of the model remain intact. The feed-forward and normalization layers ensure that the model can still learn complex representations and perform sophisticated transformations on the input data.

## Code Demonstration
[https://github.com/Zoliverling/Linear_Transformation/blob/main/Fourier_Transformer.py]("Frourier Transformer")

