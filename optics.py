import polars as pl
import numpy as np
from sklearn.cluster import OPTICS#, HDBSCAN 
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neural_network import BernoulliRBM
#from sklearn.cluster import SpectralClustering

raw = pl.read_csv("./dataset.csv", dtypes={"s1": pl.Float32, "s2": pl.Float32})
#raw = raw.filter((pl.col("open") == 1) | (pl.col("close") == 1))
signals_df = raw.select(pl.col(["s1","s2"]))
signal = signals_df.to_numpy()
grouped = signal.reshape(int(signal.shape[0]/6),6,2)
signal = np.asarray([x.mean(axis = 0) for x in grouped])

db = OPTICS(min_cluster_size=150)
classes = np.repeat(db.fit_predict(signal), repeats = 6)

joined = pl.concat([raw, pl.DataFrame({ "predicted": classes})], how = "horizontal")
joined.write_csv("res.csv")
print(joined.select(pl.exclude(["index", "time stamp"])).filter((pl.col("open") == 1) | (pl.col("thumb") == 1)))
