# titanic
TITANIC DATASET
It has 12 Features and 891 records
3 Features has no importance in analysis like id,s and text fields.
Null values:
Cabin has 80% null values. and needs to be dropped.
Age has some null values and has filled with KNN imputation. and converted to bins.
Embarked has 2 null values and is dropped row wise.
EDA:
Age and Fare are two continuous fields available. Age is converted to 4 bins based on quintiles.
More number of Females are survived 
more females are survived in Embarked class S but comparitively Embarked class C Female are saved more when compared to men
most of male who are not survived are elder than who are survived and is vice versa in case of female
Checked the IV value for all the features.
model:
Random Forest model gave an accuracy score of 85% and the predictions for the test passengers are recorded.

