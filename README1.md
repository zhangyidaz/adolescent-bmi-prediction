# adolescent-bmi-prediction

Code and sample data to reproduce analyses for adolescent BMI prediction.

## Quick Start

```bash
# 1) Install dependencies (Python 3.11.3)
pip install numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2 shap==0.46.0
pip install catboost==1.2.7 lightgbm==4.5.0

# 2) Run scripts (edit paths inside scripts if needed)
python src/BMIPre_1.py
python src/BMIPre_2.py
python src/BMIPre_3.py
```

## Data

Included de-identified/sample data:
- `data/BMIPre_1_2.csv`
- `data/BMIPre_3.csv`

## Methods (brief)

- **Split & validation:** 80/20 train–test split with **5×5 nested cross-validation** for hyperparameter tuning and performance estimation.  
- **Heteroscedasticity & calibration:** Breusch–Pagan test on **training out-of-fold residuals** (α = 0.05). If significant, **WLS calibration** is learned on the training data and applied once to the test set (no refitting).  
- **Stratified performance:** metrics stratified by key subgroups (e.g., sex, age groups, baseline BMI categories).  
- **Error analysis:** predicted-vs-observed with a smoothing line, Bland–Altman analysis, and error distributions.  
- **Model interpretation:** **SHAP** (summary, dependence, interactions, waterfall).

## Generated Figures

- Predicted-vs-observed scatterplot (with smoothing line)  
- **Bland–Altman** plot  
- **SHAP** plots: summary, dependence, interactions, waterfall

## Repository Layout

```
adolescent-bmi-prediction/
├─ src/
│  ├─ BMIPre_1.py
│  ├─ BMIPre_2.py
│  └─ BMIPre_3.py
├─ data/
│  ├─ BMIPre_1_2.csv
│  └─ BMIPre_3.csv
├─ README.md
├─ LICENSE
```
## License

- **Code:** GNU General Public License v3.0 (GPL-3.0)
