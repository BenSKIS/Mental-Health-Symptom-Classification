import pandas as pd
import category_encoders as ce
df = pd.read_csv('E:\Kaggle\Mental Health\mentalhealth\Dataset-Mental-Disorders.csv')
df.head()
features = df.iloc[:,1:-1]
labels = df.iloc[:,-1]
features
labels.columns
df.columns

chi2_results = {}
from scipy.stats import chi2_contingency
for feature in features.columns:
    contingency_table = pd.crosstab(features[feature],labels)
    chi2,p,dof,expected = chi2_contingency(contingency_table)
    chi2_results[feature] = {'chi2':chi2,'p-value':p}
    
chi2_results_df = pd.DataFrame(chi2_results).T
chi2_results_df
    
features_to_keep = [feature for feature, values in chi2_results.items() if values['chi2'] >= 10]
features = features[features_to_keep]
features_to_keep

efeatures = features
for f in features.columns:
    encoder = ce.OrdinalEncoder(cols = f,return_df=True)
    efeatures = encoder.fit_transform(efeatures)


onehotencoder = ce.OrdinalEncoder(cols = ['Expert Diagnose'],return_df = True)
elabel = onehotencoder.fit_transform(labels)
labels.head()
efeatures
elabel

from sklearn.model_selection import train_test_split
efeatures.shape
elabel.shape
x = efeatures
y = elabel
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=.3)

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(xtrain,ytrain.values.ravel())


from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the test set
y_pred = rf_classifier.predict(xtest)

# Calculate the accuracy
accuracy = accuracy_score(ytest, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print a classification report to see precision, recall, and F1-score
print(classification_report(ytest, y_pred))


