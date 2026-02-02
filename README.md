# TCS Stock Data – Live and Latest Analysis and Prediction

**Author:** Piyush Ramteke  
**Last Updated:** February 2026

---

## 📌 Project Overview

This project presents a **comprehensive analysis and forecasting approach** for **Tata Consultancy Services (TCS)** stock data. The dataset contains daily trading parameters such as Open, High, Low, Close prices, Volume, Dividends, and Stock Splits.

The objective is to study historical stock behaviour, identify meaningful patterns, and build predictive models that estimate future stock prices using **Machine Learning** and **Deep Learning (LSTM)** techniques with **hyperparameter optimization**.

### ✨ Key Highlights
- 📊 **25+ Technical Indicators** for comprehensive feature engineering
- 🔧 **Hyperparameter Tuning** with grid search for LSTM optimization
- 📈 **12+ Evaluation Metrics** for robust model comparison
- 🧪 **Statistical Significance Tests** (Diebold-Mariano, t-test, Wilcoxon)
- 🚀 **30-Day Future Price Forecasting**

---

## 📁 Project Structure

```
├── TCS Stock Data.ipynb              # Main Jupyter Notebook with complete analysis
├── TCS_stock_history.csv             # Historical stock data
├── TCS_stock_info.csv                # Stock information
├── TCS_stock_action.csv              # Stock actions (dividends, splits)
├── tcs_linear_regression_model.pkl   # Saved Linear Regression model
├── tcs_lstm_model.pth                # Saved PyTorch LSTM model (Original)
├── tcs_lstm_model_optimized.pth      # Saved PyTorch LSTM model (Optimized)
├── best_hyperparameters.pkl          # Best hyperparameters from tuning
├── tcs_scaler.pkl                    # Saved MinMaxScaler
├── tcs_prediction_results.csv        # Prediction results export
├── model_comparison_metrics.csv      # Model comparison metrics export
└── README.md                         # Project documentation
```

---

## 🔧 Technologies Used

| Category | Tools |
|----------|-------|
| Programming Language | Python 3.x |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| Statistical Analysis | SciPy |
| IDE | Jupyter Notebook / VS Code |

---

## 📊 Dataset Information

- **Source:** TCS Stock Market Data
- **Date Range:** 2002 - 2026 (20+ years)
- **Features:** Date, Open, High, Low, Close, Volume, Dividends, Stock Splits
- **Total Records:** ~5,500+ trading days

---

## 🚀 Workflow

### Step 1: Project Setup
- Install required libraries (pandas, numpy, matplotlib, seaborn, scikit-learn, torch, scipy)
- Import dependencies and configure environment

### Step 2: Data Loading
- Load CSV file containing TCS stock history
- Convert Date column to datetime format
- Sort dataset in chronological order

### Step 3: Data Preprocessing
- Check and handle missing values
- Detect outliers using IQR method
- Apply forward-fill for missing data

### Step 4: Exploratory Data Analysis (EDA)
- Price trend analysis (Open, High, Low, Close)
- Volume analysis
- Dividends and Stock Splits visualization
- Correlation heatmap
- Moving Averages (30-day, 50-day, 200-day)
- Daily percentage change distribution

### Step 5: Feature Engineering

#### 5.1 Basic Features
- Extract date features (Year, Month, Day, Weekday, Quarter)
- Create lag features (Prev_Close, Prev_Open, Prev_Volume)
- Price range features
- Volatility features
- Trading signals based on MA crossover

#### 5.2 Advanced Technical Indicators (NEW!)

| Indicator | Description | Use Case |
|-----------|-------------|----------|
| **RSI (14, 21)** | Relative Strength Index | Overbought/Oversold detection |
| **MACD** | Moving Average Convergence Divergence | Trend direction & momentum |
| **Bollinger Bands** | Volatility bands with upper/lower limits | Price range & breakouts |
| **ATR (14, 21)** | Average True Range | Volatility measurement |
| **Stochastic Oscillator** | Momentum oscillator (%K, %D) | Short-term overbought/oversold |
| **CCI** | Commodity Channel Index | Trend identification |
| **Williams %R** | Momentum indicator | Overbought/Oversold |
| **ROC (12, 26)** | Rate of Change | Momentum measurement |
| **MFI** | Money Flow Index | Volume-weighted RSI |
| **OBV** | On-Balance Volume | Volume trend confirmation |
| **EMA (12, 26, 50)** | Exponential Moving Averages | Trend smoothing |

### Step 6: Machine Learning Model - Linear Regression
- **Model:** Linear Regression
- **Features:** Open, High, Low, Volume, Prev_Close, Month, Weekday
- **Target:** Close Price
- **Metrics:** MSE, MAE, RMSE, R² Score

### Step 7: Deep Learning Model - LSTM

#### 7.1 Original LSTM Model
- **Framework:** PyTorch
- **Architecture:** 3-layer LSTM with Dropout (0.2)
- **Hidden Size:** 50
- **Sequence Length:** 60 days
- **Epochs:** 25
- **Optimizer:** Adam (lr=0.001)

#### 7.2 Hyperparameter Tuning (NEW!)
Grid search over multiple hyperparameters:

| Parameter | Values Tested |
|-----------|---------------|
| Hidden Size | 50, 100 |
| Number of Layers | 2, 3 |
| Dropout | 0.2, 0.3 |
| Learning Rate | 0.001, 0.0005 |
| Batch Size | 32, 64 |

#### 7.3 Optimized LSTM Model (NEW!)
- Uses best hyperparameters from grid search
- **Early Stopping** with patience=10
- **Learning Rate Scheduler** (ReduceLROnPlateau)
- Improved convergence and generalization

### Step 8: Comprehensive Model Evaluation (NEW!)

#### 8.1 Evaluation Metrics (12+ metrics)

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error |
| **RMSE** | Root Mean Squared Error |
| **MAE** | Mean Absolute Error |
| **MAPE** | Mean Absolute Percentage Error |
| **SMAPE** | Symmetric Mean Absolute Percentage Error |
| **R² Score** | Coefficient of Determination |
| **Explained Variance** | Proportion of variance explained |
| **Directional Accuracy** | Percentage of correct up/down predictions |
| **Theil's U** | Comparison to naive forecast (< 1 = better) |
| **Max Error** | Maximum absolute prediction error |
| **Median AE** | Median Absolute Error |
| **Correlation** | Pearson correlation coefficient |

#### 8.2 Statistical Significance Tests (NEW!)
- **Diebold-Mariano Test:** Compare forecast accuracy between models
- **Paired t-test:** Statistical comparison of absolute errors
- **Wilcoxon Signed-Rank Test:** Non-parametric alternative

### Step 9: Model Saving
- Save Linear Regression model (pickle)
- Save Original LSTM model (PyTorch .pth)
- Save Optimized LSTM model (PyTorch .pth)
- Save best hyperparameters (pickle)
- Export prediction results to CSV
- Export model comparison metrics to CSV

### Step 10: Future Price Prediction
- Predict next 30 days stock prices using optimized LSTM
- Visualize historical data with future forecasts

---

## 📈 Key Insights

1. **Long-term Growth:** TCS stock shows consistent upward growth with periodic corrections
2. **High Correlation:** Open, High, Low, and Close prices are highly correlated (>0.99)
3. **Volume Patterns:** Trading volume fluctuates heavily during major market events
4. **Moving Averages:** Effectively identify trend shifts and potential buy/sell signals
5. **Technical Indicators:** RSI and Stochastic help identify overbought/oversold conditions
6. **MACD Signals:** Useful for identifying momentum shifts and crossovers
7. **LSTM Performance:** Captures temporal patterns better than simple regression
8. **Hyperparameter Tuning:** Improves model performance significantly
9. **Directional Accuracy:** Optimized LSTM achieves better directional predictions

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+ (Python 3.11 or 3.12 recommended)
- pip package manager

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch scipy
```

### Run the Notebook

1. Open `TCS Stock Data.ipynb` in VS Code or Jupyter Notebook
2. Run all cells sequentially
3. View results and visualizations

---

## 📊 Model Performance Comparison

| Metric | Linear Regression | LSTM (Original) | LSTM (Optimized) |
|--------|-------------------|-----------------|------------------|
| MSE | ✓ | ✓ | ✓ |
| RMSE | ✓ | ✓ | ✓ |
| MAE | ✓ | ✓ | ✓ |
| MAPE (%) | ✓ | ✓ | ✓ |
| R² Score | ~0.99 | High | Highest |
| Directional Accuracy | ✓ | ✓ | Best |
| Theil's U | ✓ | ✓ | ✓ |

*Note: Actual metrics are computed and displayed in the notebook after execution.*

---

## 📉 Technical Indicators Visualization

The notebook includes comprehensive visualizations for:
- 📊 Bollinger Bands with price overlay
- 📈 RSI with overbought/oversold zones
- 📉 MACD with signal line and histogram
- 🔄 Stochastic Oscillator
- 📊 ATR (Volatility)
- 📈 CCI (Commodity Channel Index)
- 💰 Money Flow Index (MFI)
- 📊 Volume with Moving Average

---

## 🔮 Future Enhancements

- [x] ~~Implement advanced technical indicators~~ ✅ **DONE**
- [x] ~~Apply hyperparameter tuning~~ ✅ **DONE**
- [x] ~~Add comprehensive evaluation metrics~~ ✅ **DONE**
- [x] ~~Statistical significance testing~~ ✅ **DONE**
- [ ] Implement Random Forest and XGBoost models
- [ ] Use ARIMA/Prophet for time-series forecasting
- [ ] Integrate real-time stock price APIs (Yahoo Finance, Alpha Vantage)
- [ ] Build interactive dashboard using Streamlit or Power BI
- [ ] Add sentiment analysis from news data
- [ ] Implement Transformer-based models (Temporal Fusion Transformer)

---

## 📚 Technical Indicators Reference

### RSI (Relative Strength Index)
- **Range:** 0-100
- **Overbought:** > 70
- **Oversold:** < 30

### MACD (Moving Average Convergence Divergence)
- **MACD Line:** EMA(12) - EMA(26)
- **Signal Line:** EMA(9) of MACD
- **Histogram:** MACD - Signal

### Bollinger Bands
- **Middle Band:** 20-day SMA
- **Upper Band:** Middle + 2×StdDev
- **Lower Band:** Middle - 2×StdDev

### Stochastic Oscillator
- **Overbought:** > 80
- **Oversold:** < 20

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock market predictions are inherently uncertain and should not be used as financial advice. Always consult a financial advisor before making investment decisions.

---

## 📧 Contact

**Piyush Ramteke**  
Data Science & Machine Learning Enthusiast

---

## 📜 License

This project is open-source and available under the MIT License.
