# Dataset

This project uses the African Economic, Banking and Systemic Crisis dataset.

## Source

Kaggle dataset page:

https://www.kaggle.com/datasets/chirin/africa-economic-banking-and-systemic-crisis-data

Underlying global crises data are also available through the Harvard Business School Behavioral Finance and Financial Stability project:

https://www.hbs.edu/behavioral-finance-and-financial-stability/data/Pages/global.aspx

## Local File Setup

The project does not require the source CSV to be committed to GitHub.

If you want to use a local copy, place the dataset in this folder using either of these names:

```text
African_crises_dataset.csv
```

or:

```text
african_crises.csv
```

The application and notebook check for a local file first.

If no local file is available, the code attempts to load this public mirror for reproducible demonstration purposes:

```text
https://raw.githubusercontent.com/RAFrancais/Africa-economic-data-kaggle/master/african_crises.csv
```

For long-term reproducibility, keeping your own authorised local copy is preferable to depending on a third-party mirror.

## Expected Raw Structure

The commonly circulated source contains 1,059 country-year observations and the following fields:

```text
case / country_number
cc3 / country_code
country
year
systemic_crisis
exch_usd
domestic_debt_in_default
sovereign_external_debt_default
gdp_weighted_default
inflation_annual_cpi
independence
currency_crises
inflation_crises
banking_crisis
```

The project code normalises the alternative `case`/`country_number` and `cc3`/`country_code` column names automatically.

## Repository Policy

CSV files in this folder are ignored by Git through `.gitignore`.
