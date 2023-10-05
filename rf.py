import polars as pl
import polars.selectors as cs
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import m2cgen as m2c
import micromlgen

def class_t(t):
    t = list(t)
    if t == [0, 0, 0]:
        return 0
    elif t == [1, 0, 0]:
        return 2
    elif t == [0, 1, 0]:
        return 2
    elif t == [0, 0, 1]:
        return 2
    elif t == [1, 1, 0]:
        return 2
    elif t == [0, 1, 1]:
        return 2
    elif t == [1, 0, 1]:
        return 2
    else:
        return 1

raw = pl.read_csv("./dataset3.csv")
x_df = raw.select(cs.matches("EMG."))
x_df = x_df.apply(lambda r: np.mean(r))
x = (x_df.to_numpy() )
#x = x_df.to_numpy()
y_df = raw.select(pl.exclude(["TIME", "^EMG.$"]))
#print(y_df.head(5))
y_df = y_df.apply(class_t)
# y_df = y_df.select(((pl.col("open") == 0) & (pl.col("closed") == 0) & (pl.col("thumbs up") == 0)).alias("others"), pl.all())
y = y_df.to_numpy().transpose()[0]
xtr, xt, ytr, yt = train_test_split(x, y, test_size=0.2, random_state=32)

#SC = StandardScaler()
#xtr = SC.fit_transform(xtr)
#xt = SC.transform(xt)

rf = DecisionTreeClassifier() 

rf.fit(xtr, ytr)
yp = rf.predict(xt)
print(yp)
accuracy = accuracy_score(yt, yp)
print("accuracy", accuracy)
#code = micromlgen.port(rf)
#print(code)
