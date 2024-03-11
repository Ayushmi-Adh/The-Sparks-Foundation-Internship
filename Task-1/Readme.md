# Student Score Prediction using Linear Regression

## Introduction
This project utilizes the Linear Regression supervised machine learning algorithm to predict a student's percentage based on the number of hours they have studied. The primary goal is to understand and showcase the implementation of linear regression for score prediction.

**Author:** Ayushmi Adhikari


## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Steps Involved](#steps-involved)
- [Results and Visualizations](#results-and-visualizations)
- [Metrics and Evaluation](#metrics-and-evaluation)
- [Conclusion](#conclusion)

## Project Overview
This project is part of [The Sparks Foundation - Data Science & Business Analytics Internship](https://www.thesparksfoundationsingapore.org/). The primary question addressed is: "What will be the predicted score if a student studies for 9.25 hrs/day?"

## Dataset
- **Source:** [Student Scores Dataset](http://bit.ly/w-data)
- **Features:**
  - Hours Studied
  - Percentage Score

## Project Structure
- **`student_score_prediction.ipynb`:** Jupyter Notebook containing the entire project code.
- **`README.md`:** Documentation summarizing the project.


## Steps Involved
1. **Importing Libraries and Loading Data:**
   - Utilized pandas, numpy, and matplotlib to load and explore the dataset.

2. **Visualizing the Dataset:**
   - Created scatter plots to understand the distribution of scores with respect to hours studied.

3. **Data Preparation and Splitting:**
   - Extracted independent and dependent variables and split the data into training and testing sets.

4. **Training the Linear Regression Model:**
   - Utilized the scikit-learn library to train the linear regression model.

5. **Making Predictions and Evaluating Metrics:**
   - Predicted scores for a hypothetical scenario and evaluated the model using metrics such as MAE, MSE, and R-squared.

6. **Visualizing Results:**
   - Plotted the regression line, actual vs. predicted values, and a learning curve for model evaluation.

## Results and Visualizations
1. **Scatter Plot:**
   - Showed a positive correlation between hours studied and percentage scores.

2. **Regression Line Plot:**
   - Displayed the best-fit line indicating the linear relationship between hours studied and scores.

3. **Learning Curve:**
   - Visualized how the model's performance evolves with the addition of training examples.

## Metrics and Evaluation
- **Mean Absolute Error (MAE):** 4.18
- **Mean Squared Error (MSE):** 21.60
- **R-squared:** 0.945

## Conclusion
The linear regression model demonstrates a strong performance in predicting student scores based on study hours. The visualizations and metrics provide insights into the model's accuracy and generalization capabilities. Further exploration and refinement could enhance the model's predictive power.


