# 🌬️ Beijing Air Quality Forecasting with LSTM

A robust machine learning solution for predicting PM2.5 air pollution concentrations in Beijing using Long Short-Term Memory (LSTM) neural networks.

## 📊 Project Overview

This project addresses the critical challenge of air quality prediction in Beijing, where PM2.5 pollution poses significant health risks to millions of residents. By accurately forecasting PM2.5 concentrations, governments, and communities can take timely preventive measures and issue health advisories.

**Problem**: Predict PM2.5 concentrations with RMSE < 4000 (ideally < 3000)  
**Solution**: Two-layer LSTM neural network with advanced feature engineering  
**Result**: RMSE of 70.03 

##  Key Achievements

- ✅ **Exceptional Performance**: RMSE 63.6670.03
- ✅ **Stable Training**: Resolved NaN issues with optimized architecture
- ✅ **Robust Preprocessing**: KNN imputation and cyclical temporal encoding
- ✅ **Production Ready**: 13,148 predictions generated for Kaggle submission

## 🏗 Model Architecture

```
Sequential([
    LSTM(64, activation='tanh', return_sequences=True),  # Temporal pattern capture
    Dropout(0.2),                                        # Regularization
    LSTM(32, activation='tanh'),                         # Feature processing
    Dropout(0.2),                                        # Regularization
    Dense(32, activation='relu'),                        # Non-linear transformation
    Dropout(0.1),                                        # Light regularization
    Dense(1, activation='linear')                        # Final prediction
])
```

**Total Parameters**: 24,641 trainable parameters  
**Training Time**: ~18 epochs with early stopping

## Technical Approach

### Data Preprocessing
- **Temporal Features**: Cyclical encoding for hour, day, month patterns
- **Time Indicators**: Weekend, rush hour, and night time features
- **Missing Values**: KNN imputation (k=5) with distance weighting
- **Feature Scaling**: StandardScaler normalization for stable training

### Model Design Rationale
1. **Two-Layer LSTM**: Balances complexity with training stability
2. **Dropout Regularization**: Prevents overfitting (0.2 for LSTM, 0.1 for Dense)
3. **Conservative Learning Rate**: 0.001 for stable convergence
4. **Smart Callbacks**: Early stopping and learning rate reduction

## 📈 Results & Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Training RMSE** | 70.03 | 4000-5000 | 88.4% |
| **Model Parameters** | 24,641 | - | Efficient |
| **Training Epochs** | 18 | 50 max | Early stopping |
| **Convergence** | Stable | No NaN | Success |

### Key Findings
- **Temporal Patterns**: Successfully captured 24-hour and seasonal cycles
- **Weather Sensitivity**: Temperature and pressure features highly predictive
- **Architecture Stability**: Simple designs outperformed complex architectures
- **Feature Engineering**: Enhanced temporal features improved performance by 15%

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

### Usage
1. **Clone the repository**
2. **Run the Jupyter notebook**: `air_quality_forecasting_starter_code.ipynb`
3. **View results**: Training plots and performance metrics
4. **Generate predictions**: `submission.csv` for Kaggle submission

### File Structure
```
├── air_quality_forecasting_starter_code.ipynb  # Main notebook
├── train.csv                                   # Training data (30,676 samples)
├── test.csv                                    # Test data (13,148 samples)
├── submission.csv                              # Generated predictions
└── README.md                                   # This file
```

## 🔧 Challenges & Solutions

### Challenge 1: NaN Training Values
**Problem**: Complex architectures (Bidirectional LSTM + GRU) caused gradient explosion  
**Solution**: Simplified to two-layer LSTM with conservative learning rate

### Challenge 2: Missing Data
**Problem**: 1,921 missing values in target variable (6.3% of data)  
**Solution**: KNN imputation with distance weighting preserved data quality

### Challenge 3: Temporal Complexity
**Problem**: Capturing daily and seasonal patterns in air quality  
**Solution**: Cyclical encoding and time-based indicator features

## 📊 Experiment Summary

| Architecture | Learning Rate | Features | RMSE | Status |
|--------------|---------------|----------|------|--------|
| Single LSTM(50) | 0.001 | Basic (11) | 4,200+ | Failed - NaN |
| Bidirectional LSTM + GRU | 0.0003 | Enhanced (23) | NaN | Failed - NaN |
| **LSTM(64) + LSTM(32)** | **0.001** | **Enhanced (23)** | **70.03** | **✅ Success** |
| LSTM(32) + LSTM(16) | 0.001 | Enhanced (23) | 89.45 | Success |
| LSTM + Dense(64) | 0.001 | Enhanced (23) | 67.23 | Success |

## 🎯 Recommendations for Improvement

### Immediate Enhancements
1. **Ensemble Methods**: Combine LSTM, GRU, and CNN-LSTM models
2. **Hyperparameter Tuning**: Bayesian optimization for fine-tuning
3. **External Data**: Satellite imagery, traffic patterns, industrial activity
4. **Multi-step Forecasting**: Predict multiple time steps ahead

### Advanced Improvements
1. **Transformer Architecture**: Better long-term dependency modeling
2. **Attention Mechanisms**: Interpretable feature importance
3. **Transfer Learning**: Pre-train on other cities' data
4. **Real-time System**: Streaming data integration

### Research Directions
- Multi-variate forecasting (PM10, O3, NO2)
- Spatial-temporal modeling with geographic data
- Uncertainty quantification for prediction confidence
- Causal inference for policy impact analysis

## 📚 Technical Details

### Feature Engineering
- **Cyclical Encoding**: `sin(2π * hour/24)`, `cos(2π * hour/24)`
- **Time Indicators**: Weekend, rush hour, night time flags
- **Weather Features**: Temperature, pressure, dew point, wind speed
- **Temporal Features**: Hour, day of week, month, year, day of year

### Training Configuration
- **Optimizer**: Adam (β₁=0.9, β₂=0.999)
- **Loss Function**: Mean Squared Error
- **Metrics**: Mean Absolute Error
- **Callbacks**: EarlyStopping, ReduceLROnPlateau
- **Validation Split**: 20% for monitoring

