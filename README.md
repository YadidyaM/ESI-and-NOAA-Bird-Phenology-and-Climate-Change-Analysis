# Bird Phenology and Climate Change Analysis

**With CPU I was able to achieve a Test Accuracy of 77.17%. The model is working perfectly and predicts the classes of the image with 90% Accuracy. Use GPU and run more epochs.**

## Overview

This project investigates the impact of climate change on bird migration patterns through a combination of image classification using deep learning and predictive modeling with climate and biological data. It involves training a deep neural network for bird species classification and analyzing climate data to predict migration start dates.

## Key Components

### 1. Deep Learning Model for Bird Classification
- **Objective**: To classify bird species using a pre-trained MobileNet model to segregate to batches as I have 80k images.
- **Process**:
  - Load a pre-trained MobileNet model from a specified path.
  - Prepare data using `ImageDataGenerator` for training, validation, and testing.
  - Evaluate the model's performance on a test dataset.
  - Fine-tune the model by unfreezing and training some layers with a lower learning rate.
  - Save the fine-tuned model for future use.
- **Keywords**: MobileNet, Transfer Learning, Image Classification, Data Augmentation, Model Evaluation, Fine-Tuning, TensorFlow, Keras.

### 2. Predictive Modeling of Bird Migration Patterns
- **Objective**: To predict bird migration start dates based on climate data.
- **Process**:
  - Load and preprocess climate and bird tracking data.
  - Merge datasets and prepare features for modeling.
  - Apply machine learning models such as Random Forest, Gradient Boosting, and XGBoost to predict migration start dates.
  - Fine-tune models using Grid Search and evaluate their performance based on metrics like Mean Squared Error (MSE) and R-squared (R2).
- **Keywords**: Predictive Modeling, Random Forest, Gradient Boosting, XGBoost, Hyperparameter Tuning, Feature Importance, Climate Data Analysis, Regression Analysis.

### 3. Advanced Data Visualization
- **Objective**: To visually explore and interpret the relationships between climate variables and bird migration patterns.
- **Process**:
  - Generate feature importance plots to understand the impact of different features.
  - Create partial dependence plots to explore the effect of specific features on predictions.
  - Use pair plots to visualize relationships between key variables.
  - Plot time-series data to examine trends over time.
- **Keywords**: Data Visualization, Feature Importance, Partial Dependence Plots, Pair Plots, Time-Series Analysis, Seaborn, Matplotlib.

  ## Dataset Citations and Credits

- **eBird**: Data provided by eBird, a citizen science project managed by the Cornell Lab of Ornithology. For more information, visit [eBird](https://ebird.org).

- **NOAA Climate Data Online (CDO)**: Climate data provided by the National Oceanic and Atmospheric Administration (NOAA) through their Climate Data Online (CDO) service. For more information, visit [NOAA Climate Data Online](https://www.ncdc.noaa.gov/cdo-web/).

## Acknowledgements

Special thanks to the eBird and NOAA CDO teams for providing the data used in this analysis.

## Expected Outcomes
- **Enhanced Bird Classification Model**: Improved accuracy in identifying bird species with a fine-tuned MobileNet model.
- **Predictive Insights**: Insights into how climate variables influence bird migration patterns using advanced regression models.
- **Visual Interpretations**: Comprehensive visualizations that aid in understanding the relationship between climate and migration trends.

## Applications
- **Ecology**: Understanding how climate change affects bird migration, which can inform conservation strategies.
- **Climate Science**: Providing empirical evidence of climate impacts on wildlife.
- **Machine Learning**: Demonstrating practical applications of deep learning and machine learning in environmental science.

This project bridges machine learning techniques with environmental data to address critical questions in climate impact analysis and biodiversity conservation.
