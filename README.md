# TCS Stock Data – Live and Latest Analysis and Prediction

**Author:** Piyush Ramteke

---

## 📌 Project Overview

This project presents a complete analysis and forecasting approach for **Tata Consultancy Services (TCS)** stock data. The dataset contains daily trading parameters such as Open, High, Low, Close prices, Volume, Dividends, and Stock Splits.

The objective is to study historical stock behaviour, identify meaningful patterns, and build predictive models that estimate future stock prices using **Machine Learning** and **Deep Learning (LSTM)** techniques.

---

## 📁 Project Structure

```
├── TCS Stock Data.ipynb          # Main Jupyter Notebook with complete analysis
├── TCS_stock_history.csv         # Historical stock data
├── TCS_stock_info.csv            # Stock information
├── TCS_stock_action.csv          # Stock actions (dividends, splits)
├── tcs_linear_regression_model.pkl   # Saved Linear Regression model
├── tcs_lstm_model.pth            # Saved PyTorch LSTM model
├── tcs_scaler.pkl                # Saved MinMaxScaler
├── tcs_prediction_results.csv    # Prediction results export
└── README.md                     # Project documentation
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
| IDE | Jupyter Notebook / VS Code |

---

## 📊 Dataset Information

- **Source:** TCS Stock Market Data
- **Date Range:** 2002 - 2024 (20+ years)
- **Features:** Date, Open, High, Low, Close, Volume, Dividends, Stock Splits
- **Total Records:** ~4,400+ trading days

---

## 🚀 Workflow

### Step 1: Project Setup
- Install required libraries
- Import dependencies

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
- Extract date features (Year, Month, Day, Weekday, Quarter)
- Create lag features (Prev_Close, Prev_Open, Prev_Volume)
- Price range features
- Volatility features
- Trading signals based on MA crossover

### Step 6: Machine Learning Model
- **Model:** Linear Regression
- **Features:** Open, High, Low, Volume, Prev_Close, Month, Weekday
- **Target:** Close Price
- **Metrics:** MSE, MAE, R² Score

### Step 7: Deep Learning Model (LSTM)
- **Framework:** PyTorch
- **Architecture:** 3-layer LSTM with Dropout
- **Sequence Length:** 60 days
- **Epochs:** 25
- **Optimizer:** Adam

### Step 8: Model Evaluation
- Compare Linear Regression vs LSTM performance
- Visualize actual vs predicted prices

### Step 9: Model Saving
- Save models using pickle and PyTorch
- Export prediction results to CSV

### Step 10: Future Price Prediction
- Predict next 30 days stock prices

---

## 📈 Key Insights

1. **Long-term Growth:** TCS stock shows consistent upward growth with periodic corrections
2. **High Correlation:** Open, High, Low, and Close prices are highly correlated (>0.99)
3. **Volume Patterns:** Trading volume fluctuates heavily during major market events
4. **Moving Averages:** Effectively identify trend shifts and potential buy/sell signals
5. **LSTM Performance:** Captures temporal patterns better than simple regression

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+ (Python 3.11 or 3.12 recommended for TensorFlow compatibility)
- pip package manager

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch
```

### Run the Notebook

1. Open `TCS Stock Data.ipynb` in VS Code or Jupyter Notebook
2. Run all cells sequentially
3. View results and visualizations

---

## 📊 Model Performance

| Model | MSE | MAE | RMSE | R² Score |
|-------|-----|-----|------|----------|
| Linear Regression | Low | Low | Low | ~0.99 |
| LSTM | Varies | Varies | Varies | - |

*Note: Actual metrics depend on the data split and training run.*

---

## 🔮 Future Enhancements

- [ ] Implement Random Forest and XGBoost models
- [ ] Apply hyperparameter tuning
- [ ] Use ARIMA/Prophet for time-series forecasting
- [ ] Integrate real-time stock price APIs (Yahoo Finance, Alpha Vantage)
- [ ] Build interactive dashboard using Streamlit or Power BI
- [ ] Add sentiment analysis from news data

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
