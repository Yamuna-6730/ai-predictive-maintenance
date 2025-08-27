# AI Predictive Maintenance

## 📝 Project Overview
Unexpected failures in energy infrastructure components like **turbines, transformers, and solar panels** lead to **loss of energy and high maintenance costs**.  
This project applies **AI and Machine Learning** techniques to predict **Remaining Useful Life (RUL)** of engines, helping to schedule **predictive maintenance** before failures occur.  

**Problem Statement:**  
- Predict how many operational cycles remain before an engine fails.  
- The data is multivariate time series from multiple engines (fleet data).  
- Each engine starts with unknown initial wear; faults develop over time.  
- Training data runs engines until failure; test data stops before failure.  

**Proposed Solution:**  
- Train ML models (baseline and deep learning) on sensor and operational data.  
- Use time-series modeling to predict RUL for each engine.  
- Output alerts like: “Engine X likely to fail in Y cycles”.  
- Helps reduce downtime and maintenance costs.

---

This project uses the **CMAPSS Turbofan Engine Degradation Simulation Dataset** for building machine learning models to predict the **Remaining Useful Life (RUL)** of engines.  
The dataset consists of four subsets: **FD001, FD002, FD003, FD004**.

---

## 📂 Folder Structure
```

 ai_predictive_maintenance/
├── data/ 
│ ├── raw/ # Original CMAPSS dataset files (train, test, RUL)
│ └── processed/ # Cleaned CSVs with RUL column added
│
├── notebooks/ 
│ ├── 01_data_preprocessing.ipynb #  Load, clean, RUL, EDA
│ ├── 02_baseline_models.ipynb #  ML baseline models
│ ├── 03_deep_learning_models.ipynb #  LSTM/TCN models
│ ├── 04_dashboard_and_outputs.ipynb # Dashboard & visualization
│
├── utils/ 
│ ├── data_pipeline.py # Functions to load & clean data
│ ├── feature_engineering.py # Feature extraction functions
│ └── visualization.py # Plotting utilities
│
├── outputs/ 
│ ├── models/ # Trained models (.pkl, .pt)
│ └── figures/ # Plots and charts
│
└── README.md # Project documentation

```

---

## 📊 Week 1 Progress – Data Preprocessing & EDA

1. **Loaded raw data** (`train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt`).  
2. **Assigned column names**:  
   - `unit_number`, `time_in_cycles`, 3 operational settings, and 21 sensor measurements.  
3. **Computed RUL (Remaining Useful Life)** for training data:  
   - `RUL = max_cycle_per_engine - current_cycle`.  
4. **Saved preprocessed datasets** as CSV files in `data/processed/`.  
5. **EDA (Exploratory Data Analysis)**:  
   - RUL distribution histograms.  
   - Sensor signal plots for individual engines (e.g., Engine 1).  

---

## ✅ Outputs (Week 1)

- **Processed CSVs**:  
  - `train_FD001.csv`, `test_FD001.csv`, `rul_FD001.csv`  
  - `train_FD002.csv`, … up to FD004  
- **Figures** (in `outputs/figures/`):  
  - `rul_distribution_FD001.png`  
  - `engine1_sensors.png`  

---


