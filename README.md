# music-prediction-algo
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
music_list = pd.read_csv("music.csv")
x = music_list.drop(columns=['genre'])
y = music_list['genre']
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3)
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
score = accuracy_score(y_test, predictions)
score
