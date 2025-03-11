from sklearn  import (cluster,
datasets,
decomposition,
discriminant_analysis,
dummy,
ensemble,
feature_selection as ftr_sel,
linear_model,
metrics,
model_selection as skms,
multiclass as skmulti,
naive_bayes,
neighbors,
pipeline,
preprocessing as skpre,
svm,
tree)
#import mlwpy.py
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import display
#%matplotlib inline
iris= datasets.load_iris ()
iris_df= pd.DataFrame (iris.data,columns=iris.feature_names)
iris_df ['target']= iris.target
display(pd.concat([iris_df.head (3),iris_df.tail(3)]))
#iris_p=sns.load_dataset(iris)
#sns.pairplot (iris_df,hue='target')#, height=1.5
sns.pairplot(iris_df,hue='target')
plt.show()
print('targets: {}'.format(iris.target_names), iris.target_names[0], sep="\n")
(iris_train_ftrs, iris_test_ftrs,iris_train_tgt, iris_test_tgt) = skms.train_test_split(iris.data,iris.target, test_size=.25)
print("Train features shape:", iris_train_ftrs.shape)
print("Test features shape:", iris_test_ftrs.shape)
# за замовчуванням n neighbors = 5
knn= neighbors.KNeighborsClassifier(n_neighbors=3)
fit = knn.fit(iris_train_ftrs, iris_train_tgt)
preds= fit.predict(iris_test_ftrs)
#порівняння передбачень з відкладеними тестовими цілями
print("3NN accuracy:",metrics.accuracy_score(iris_test_tgt, preds))
print('The data matrix:\n',iris['data'])
print('The classification target:\n',iris['target'])
print('The names of the dataset columns:\n',iris['feature_names'])
print('The names of target classes:\n',iris['target_names'])
print('The full description of the dataset:\n',iris['DESCR'])
print('The path to the location of the data:\n',iris['filename'])
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print("Прогноз: {}".format(prediction))
print("Спрогнозированная метка: {}".format(iris['target_names'][prediction]))