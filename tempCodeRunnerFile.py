import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest,f_classif

df=pd.read_csv("Churn_Modelling.csv")

#pca=PCA(n_components=5)

x=df.drop(columns=['Exited'])
y=df['Exited']

x=pd.get_dummies(x,drop_first=True)
#x=pca.fit_transform(x)
selector=SelectKBest(score_func=f_classif,k=5)
x_new=selector.fit_transform(x,y)
selected_features= x.columns[selector.get_support()]
print("Selected features : ")
for i in selected_features:
    print(i)


x_train,x_test,y_train,y_test= train_test_split(x_new,y,test_size=0.2,random_state=42)

lr = LogisticRegression()
lr.fit(x_train,y_train)
probs= lr.predict_proba(x_test)[:,1]

print("Predicted Probabilities: ")
for i,prob in enumerate(probs[:10]):
    print(f"Customer {i+1} : {prob}")
    
print(roc_auc_score(y_test,probs))
