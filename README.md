# Credit_Risk_Analysis
 Assessing individual's credit default risk with pandas, scikit-learn, and various data sampling techniques. The goal of the project is to  develop a model for predicting individual risk of default on outstanding loans. 

 Models were analyzed on accuracy scores, precision, recall, and F1 scores. 


## Oversampling

### Random Oversampling - model 1

Our data is imbalanced so let's start with oversampling the high risk loans. 

![Random Oversampling Confusion Matrix]("Resources/ros_cm.png")

![Random Oversampling Classification Report]("Resources/ros_cr.png")

* Balanced accuracy score of 0.647

The balanced accuracy score from the random oversampling of the dataset is lower than desired. A 0.647 accuracy score indicates the model is inaccurately classifying a significant number of outstanding loans. 

Precision:
* High Risk = 0.01
* Low Risk = 1.00

A precision score of essentially 0 tells us that the random oversampling method is not doing a good job of predicting high risk loans which are actually high risk (true positives).

The model over-assigns a high risk classification to loans which are actually low-risk (false positives). While the model is doing a poor job of determining true positives, a higher number of false positives may not be an indication of a bad model for assessing risk. Higher false positives mean the model is more conservative and may help with the lender's risk management. 

Additionally, it is important to remember that both precision and recall cannot be maximized by the same model. Higher precision leads to lower recall and higher recall leads to lower precision. IF we are attempting to maximize recall we will inherently see a lower precision score. 

Recall:
* High Risk = 0.69
* Low Risk = 0.60

Recall is likely a better measure of success for this model. Recall measures how well a model is correctly predicting positive outcomes. In our case, positive outcomes are high risk loans. Therefore, because we are trying to develop a model that is accurately predicting default risk, recall is a better measure of model accuracy. 

The 0.69 recall for high risk loans indicates the model has about a 70% chance of classifying high risk loans. The recall score is good, but improvements would help the lender manage future default risks. 

F1 Score:
* High Risk = 0.02
* Low Risk = 0.75

F1 is low for high risk classifications. This may not indicate inaccuracy because the F1 score accounts for both precision and recall. Therefore, a large difference in the precision and recall scores can lead to low F1 scores. Additionally, when analyzing balanced data F1 scores are less relevant than mis-classification of our target metric. 

### SMOTE Oversampling - model 2

Will using the SMOTE algorithm for oversampling improve our model?

![SMOTE Oversampling Confusion Matrix]("Resources/smote_cm.png")

![SMOTE Oversampling Classification Report]("Resources/smote_cr.png")

* Balanced accuracy score of 0.662

The accuracy score slightly improved using the SMOTE  algorithm to oversample the high-risk credit data. Still, the score is not high enough to suggest confidence in using the model for predicting future default risks. 

Precision:
* High Risk = 0.01
* Low Risk = 1.00

No significant changes from the random oversampling model. Still a significant amount of false positives indicating conservative default predictions. 

Recall:
* High Risk = 0.63
* Low Risk = 0.69

Lower recall for high risk loans indicates the SMOTE model was worse at predicting actual high risk outstanding loans. The model assigned a low risk score to more loans that were actually high risk than the random oversampling model making it less useful for our default analysis. 

F1 Score:
* High Risk = 0.02
* Low Risk = 0.82

The SMOTE did not improve the F1 score for high risk loans. This outcome is not unexpected given balanced data.

## Undersampling

What if we undersample the low risk loans?

### ClusterCentroids Undersampling- model 3

![ClusterCentroids Undersampling Confusion Matrix]("Resources/cc_cm.png")

![ClusterCentroids Undersampling Classification Report]("Resources/cc_cr.png")

* Balanced accuracy score of 0.544

The accuracy score dropped significantly when undersampling. Because we undersampled the low risk data this may indicate the model is worse at predicting high risk data than we previously determined by over sampling. 

Precision:
* High Risk = 0.01
* Low Risk = 1.00

No significant changes from the oversampling models. Still a significant amount of false positives indicating conservative default predictions and we are trying to improve recall so low precision is less of a drawback. 

Recall:
* High Risk = 0.69
* Low Risk = 0.63

We see an improvement in recall from the SMOTE algorithm, but no improvement over the random oversampling model. This most likely indicates the SMOTE algorithm was susceptible to outliers in the dataset. Overall, recall is still good but may be improved upon further using different modeling techniques.

F1 Score:
* High Risk = 0.01
* Low Risk = 0.57

ClusterCentroid undersampling did not improve the F1 score for high risk loans. This outcome is not unexpected given balanced data.

## Combination Sampling Techniques

### SMOTEEN Algorithm - model 4

What if we combine SMOTE with a nearest neighbors algorithm? 

First we oversample with SMOTE and then clean the data by dropping data points where nearest neighbors are from a different class. This model should help overcome SMOTE's potential to be affected by outliers. 

![SMOTEEN Combination Sampling Confusion Matrix]("Resources/smoteen_cm.png")

![SMOTEEN Combination Sampling Classification Report]("Resources/smoteen_cr.png")

* Balanced accuracy score of 0.678

The accuracy score improved to the highest score of our sampling models, slightly improving the SMOTE accuracy score. Higher accuracy could indicate SMOTEEN is a better predictor of high credit risk. 

Precision:
* High Risk = 0.01
* Low Risk = 1.00

No significant changes from the other sampling models. Still a significant amount of false positives indicating conservative default predictions and we are trying to improve recall so low precision is less of a drawback. 

Recall:
* High Risk = 0.78
* Low Risk = 0.57

We see an significant improvement in recall up to 0.78 from a previous high of 0.69. The higher recall score indicates the SMOTEEN model is better at classifying true positives when analyzing the risk level of outstanding loans. 

F1 Score:
* High Risk = 0.02
* Low Risk = 0.73

SMOTEEN also improved the F1 score for high risk loans. This is a positive outcome given the improvement is likely due to higher recall. It is also possible the F1 score was affected by the testing data being unbalanced compared to the previously used over and undersampling methodologies. 


