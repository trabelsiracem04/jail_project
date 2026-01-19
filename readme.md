Exploratory Data Analysis and Modeling

This project involved working with a large, complex, and noisy real-world dataset related to criminal cases and sentencing outcomes.

A significant part of the work focused on exploratory data analysis (EDA) in order to:

understand the structure of the data across multiple stages (initiation, disposition, sentencing)

identify missing values, inconsistent fields, and mixed data types

analyze the distribution of sentence lengths, which is highly skewed and sparse

understand how legal variables (offense category, charge class) differ from typical ML features

The EDA revealed that the dataset is not clean or linear, and that sentencing outcomes depend on many factors that are difficult to capture purely from tabular data.

Modeling Approach

I initially trained a Random Forest regression model to predict jail sentence length in days.
Despite careful preprocessing and feature engineering, the model achieved limited performance, especially for longer or uncommon sentences.

This result highlighted an important limitation:

Complex legal and judicial outcomes are not easily predictable using standard machine learning models and tabular features alone.

To further explore this, I experimented with more advanced tree-based models such as:

XGBoost

CatBoost

While these models provided some improvements, overall performance remained constrained by:

high variance in sentencing decisions

unobserved legal and contextual factors

strong imbalance and skewness in the target variable

Key Learnings

This project was valuable even though the predictive performance was limited. Through this work, I learned how to:

perform EDA on a difficult, real-world dataset

clean and preprocess highly categorical legal data

handle skewed regression targets

compare multiple machine learning models realistically

understand when machine learning is not the right tool for strong prediction

This project reinforced the idea that model performance depends as much on data quality and problem structure as on the choice of algorithm.

Important Note

This project is for educational and exploratory purposes only.
It is not intended to be used for legal, judicial, or policy decision-making.
thank you
