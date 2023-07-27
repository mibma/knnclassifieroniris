import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
class KNNClassifier:
    def __init__(self,k=5):
        self.k=k


    def fit(self,X,y):
        self.X_train=X
        self.y_train=y

    def euclidean_dist(self,x1,x2):
        return np.sqrt(np.sum(x1-x2)**2)
    def predict(self,X):
        y_pred= [self._predict_single(x) for x in X]
        return np.array(y_pred)
    def _predict_single(self,x):
        distances=[self.euclidean_dist(x,x_train) for x_train in self.X_train]

        k_indices=np.argsort(distances)[:self.k]

        k_nearest_labels= [self.y_train[i] for i in k_indices]

        most_common =np.bincount(k_nearest_labels).argmax()

        return most_common


dataset=pd.read_csv('C:\\Users\\HP\\Desktop\\pandas\\iriscsv.csv')
column_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

# Assigning the list of column names to DataFrame columns
dataset.columns = column_names
print(dataset.head())

model=KNNClassifier(k=5)
X=dataset[['SepalLength','SepalWidth','PetalLength','PetalWidth']].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dataset['Species'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train,y_train)
prediction=model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print("Accuracy:", accuracy)
#accuracy=0.8333333333333334
