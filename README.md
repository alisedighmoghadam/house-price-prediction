<!-- Python -->

# House Price Prediction

Predicting house prices from living area and number of bedrooms, comparing scikit-learn's closed-form **Linear Regression** with a gradient-based **SGD Regressor**.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

## Dataset

`house_price.csv` – 47 houses with three columns:

| Column | Description |
| --- | --- |
| `size` | Living area (sq ft) |
| `bedroom` | Number of bedrooms |
| `price` | Sale price (USD) – target |

Correlation with price: **size 0.85**, bedrooms 0.44. No missing values.

## Approach

1. Exploratory analysis with Seaborn regression plots and a correlation matrix.
2. 70/30 train/test split (`random_state=42`).
3. **Model 1 – `LinearRegression`** on the raw features.
4. **Model 2 – `SGDRegressor`** on standardised features (`StandardScaler`).
5. Evaluate with MAE, MSE, RMSE and MAPE, and plot actual vs. predicted prices.

## Results

| Model | MAE | RMSE | MAPE |
| --- | --- | --- | --- |
| Linear Regression | $68,296 | $87,242 | 17.08% |
| SGD Regressor | $68,593 | $87,459 | 17.11% |

Both models converge to almost the same solution: SGD with scaled inputs reaches the closed-form optimum. Size drives most of the prediction, while the bedroom count adds little once size is known.

**On the metrics:** MAE weighs every error equally, while MSE/RMSE square the errors and therefore highlight the largest mispredictions.

## Run it

```bash
git clone https://github.com/alisedighmoghadam/house-price-prediction.git
cd house-price-prediction
pip install numpy pandas scikit-learn seaborn matplotlib jupyter
jupyter notebook house_price_prediction.ipynb
```

Built as part of DPS920 coursework at Seneca Polytechnic.
