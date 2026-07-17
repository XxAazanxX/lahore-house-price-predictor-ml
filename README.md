# lahore-house-price-predictor-ml
Introduction:

An Object-Oriented Machine Learning pipeline developed to clean data, train a Linear Regression model, and predict real estate property prices across Lahore, Pakistan. 

This project demonstrates clean, production-grade code architecture designed to transition theoretical concepts into an industrial engineering use case.

##  Project Overview & Objective
Traditional real estate valuation relies heavily on manual, subjective estimates. This repository builds a data-driven alternative. By feeding a local housing dataset into a trained regression model, the system analyzes key architectural attributes to forecast accurate market valuations in local currency.

##  Tech Stack & Key Libraries
* **Language:** Python
* **Data Engineering & Analysis:** `pandas` (for dataset cleaning, handling missing values, and column extraction)
* **Machine Learning:** `scikit-learn` (implementing `LinearRegression` models and train-test evaluation splits)
* **Environment:** VS Code

## Repository Structure & Code Architecture
Unlike generic beginner scripts, this pipeline is built using **Object-Oriented Programming (OOP)** principles to ensure code usability, modularity, and scalability:
* **Data Isolation:** Isolates independent variables (`Size_sqm`, `Bedrooms`) from the target feature (`Total_Price_Local`), entirely eliminating data leakage vulnerabilities.
* **Encapsulation:** Encloses data loading, training workflows, score evaluation, and live predictions within a singular, secure Python Class (`LahoreHousePricePredictor`).
* **Type Hinting & Safety:** Employs explicit data type expectations and handles standard file execution anomalies gracefully via comprehensive error catch pipelines.

## How It Works (Step-by-Step)
1. **`load_data`**: Reads `lahore_housing_prices.csv`, drops incomplete data fields, isolates core feature matrices, and shuffles data into an 80/20 train-test ratio.
2. **`train`**: Fits the Scikit-Learn `LinearRegression` algorithm mathematically onto your feature matrices.
3. **`evaluate`**: Scores the final validation performance using the R-squared (R²) metric to confirm overall prediction reliability.
4. **`predict_price`**: Dynamically packages custom manual parameters into safe structured DataFrames to generate real-time real estate pricing estimations.

## Engineering Takeaways
This build successfully showcases structural competency in clean backend programming, end-to-end data manipulation, and predictive analytics. It validates practical discipline in building reusable technical logic rather than flat scripts.
