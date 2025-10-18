# PROHI Dashboard - ObesityVision

![Obesity-Vision](./img/logo.svg)

## Introduction

**Obesity-Vision** is an interactive web dashboard designed to explore, visualize, and predict obesity levels based on individual lifestyle and physical condition factors. The project aims to demonstrate how data-driven tools can enhance understanding of obesity risk patterns and support preventive health strategies.

The medical problem addressed is the **rising global prevalence of obesity**, a major risk factor for chronic diseases such as diabetes, cardiovascular disorders, and metabolic syndrome. Early identification of individuals at risk is essential to guide lifestyle interventions and reduce long-term health burdens.

We used the UCI dataset [**“Estimation of Obesity Levels Based on Eating Habits and Physical Condition”**](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition), which contains anthropometric and behavioral attributes (e.g., physical activity, diet, water intake, and alcohol consumption) labeled according to obesity level.

For this dashboard, the data preprocessing and predictive modeling were performed in Python using **Scikit-Learn** and **XGBoost**, with interpretability visualizations generated using **SHAP**.

The resulting dashboard allows users to:
- Interactively explore correlations between lifestyle habits and obesity outcomes.  
- Visualize the impact of features using SHAP value interpretation.  
- Input custom values to **predict obesity level categories** with a pre-trained model.

All analyses are intended for **educational and exploratory purposes**, demonstrating the potential of machine learning to support obesity prevention and health education.

## System description

### Dependencies

Tested on Python 3.12.7 with the following packages:
  - Jupyter v1.1.1
  - Streamlit v1.46.1
  - Seaborn v0.13.2
  - Plotly v6.2.0
  - Scikit-Learn v1.7.0
  - shap v0.48.0
  - xgboost v3.0.5

### Installation

Run the commands below in a terminal to configure the project and install the package dependencies for the first time.

If you are using macOS, you may need to install **Xcode Command Line Tools** and **OpenMP (libomp)** to support XGBoost.

Check the official Streamlit documentation [Streamlit](https://docs.streamlit.io/get-started/installation/command-line#prerequisites).

1. Create the environment with `python -m venv env`
2. Activate the virtual environment for Python
   - `source env/bin/activate` [in Linux/Mac]
   - `.\env\Scripts\activate.bat` [in Windows command prompt]
   - `.\env\Scripts\Activate.ps1` [in Windows PowerShell]
3. Make sure that your terminal is in the environment (`env`) not in the global Python installation
4. Install required packages `pip install -r ./requirements.txt`
5. Check that everything is ok running `streamlit hello`

### Execution

To run the dashboard execute the following command:

``` bash
> streamlit run Welcome.py
# If the command above fails, use:
> 1. Create the environment with `python -m venv env`
  2. Activate the virtual environment for Python
    - `source env/bin/activate` [in Linux/Mac]
    - `.\env\Scripts\activate.bat` [in Windows command prompt]
    - `.\env\Scripts\Activate.ps1` [in Windows PowerShell]
  3. Make sure that your terminal is in the environment (`env`) not in the global Python installation
```

## Contributors

_Yueyan Li_

_Zsolt Fehér_

_Weiqi Kong_

_Christoffer Brändefors_

_Naznin Akhtar_