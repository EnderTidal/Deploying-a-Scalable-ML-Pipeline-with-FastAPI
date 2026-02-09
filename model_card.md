# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier developed to predict whether an individual's income exceeds $50,000 per year based on census data. The model was created as part of a machine learning deployment project using FastAPI.

**Model Information:**
- **Model Type:** Random Forest Classifier
- **Framework:** scikit-learn
- **Model Version:** 1.0.0
- **Date:** February 2026
- **Developer:** Udacity ML DevOps Project
- **Hyperparameters:**
  - n_estimators: 100
  - max_depth: 10
  - random_state: 42
  - n_jobs: -1 (parallel processing)

## Intended Use

This model is intended for educational and demonstration purposes as part of a machine learning deployment project. It predicts binary income classification (>50K or <=50K) based on demographic and employment-related features from census data.

**Primary Use Cases:**
- Educational demonstrations of ML model deployment
- Understanding income prediction based on census features
- Learning FastAPI integration with ML models

**Out-of-Scope Use Cases:**
- Real-world hiring or lending decisions
- Any use case requiring legally compliant decision-making
- Production systems without proper validation and monitoring

## Training Data

The model was trained on the Census Income dataset (also known as "Adult" dataset) from the UCI Machine Learning Repository.

**Dataset Characteristics:**
- **Source:** 1994 Census database
- **Total Samples:** 32,561 instances
- **Training Split:** 80% (26,048 samples)
- **Features:** 14 attributes including age, workclass, education, marital status, occupation, relationship, race, sex, capital gain/loss, hours per week, and native country
- **Target Variable:** Binary classification - income >50K or <=50K

**Preprocessing:**
- Categorical features were one-hot encoded using scikit-learn's OneHotEncoder
- Labels were binarized using LabelBinarizer
- No scaling was applied to continuous features (suitable for tree-based models)

**Categorical Features Used:**
- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- native-country

## Evaluation Data

The model was evaluated on a held-out test set comprising 20% of the original dataset.

**Test Set Characteristics:**
- **Size:** 6,513 samples
- **Split Method:** Random stratified split with random_state=42
- **Same preprocessing pipeline** applied as training data

## Metrics

The model's performance was evaluated using three standard classification metrics:

**Overall Performance:**
- **Precision:** 0.7962 (79.62%)
  - Of all instances predicted as >50K, 79.62% were correct
- **Recall:** 0.5372 (53.72%)
  - Of all actual >50K instances, the model correctly identified 53.72%
- **F1 Score:** 0.6416 (64.16%)
  - Harmonic mean of precision and recall

**Interpretation:**
The model demonstrates high precision, meaning when it predicts someone earns >50K, it is usually correct. However, the recall is moderate, indicating the model misses a significant portion of high-income individuals (false negatives). This trade-off may be acceptable depending on the use case.

**Slice Performance:**
Performance metrics were computed across different categorical feature slices (e.g., by education level, occupation, race, sex) and saved to `slice_output.txt`. This analysis helps identify potential biases and performance disparities across demographic groups.

## Ethical Considerations

**Bias and Fairness:**
- The model uses sensitive attributes such as race, sex, and native-country as features, which may perpetuate historical biases present in the training data
- Performance should be carefully evaluated across demographic groups to ensure fairness
- The 1994 census data may not reflect current socioeconomic conditions

**Privacy:**
- The model was trained on publicly available census data
- No personally identifiable information (PII) is used
- Predictions should not be used to make decisions about specific individuals

**Potential Harms:**
- Using this model for hiring, lending, or other consequential decisions could result in discriminatory outcomes
- The model may perform differently across demographic groups, potentially disadvantaging certain populations
- Historical biases in census data may be amplified by the model

**Recommendations:**
- Conduct thorough fairness audits before any real-world deployment
- Implement human oversight for any decisions influenced by model predictions
- Regularly monitor model performance across demographic groups
- Consider removing or carefully handling sensitive attributes

## Caveats and Recommendations

**Limitations:**
1. **Temporal Validity:** The model is trained on 1994 census data and may not generalize well to current economic conditions
2. **Geographic Scope:** Primarily focused on U.S. census data and may not apply to other countries
3. **Class Imbalance:** The dataset has significantly more instances of <=50K than >50K, which affects model performance
4. **Feature Limitations:** The model relies on a limited set of features and may miss important factors affecting income

**Recommendations:**
1. **Regular Retraining:** Update the model with more recent census data to maintain relevance
2. **Fairness Monitoring:** Continuously monitor performance across demographic groups and address disparities
3. **Ensemble Methods:** Consider combining multiple models to improve recall while maintaining precision
4. **Feature Engineering:** Explore additional features or interactions that may improve predictive power
5. **Threshold Tuning:** Adjust the classification threshold based on the specific use case requirements (precision vs. recall trade-off)
6. **Validation:** Conduct extensive validation on recent data before any production deployment
7. **Human-in-the-Loop:** Always include human review for decisions based on model predictions

**Model Maintenance:**
- Monitor model performance over time for drift
- Retrain periodically with updated data
- Track prediction distributions to detect anomalies
- Maintain version control for reproducibility
