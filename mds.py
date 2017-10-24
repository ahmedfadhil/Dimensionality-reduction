# MDS
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.datasets import load_breast_cancer
from adspy_shared_utilities import plot_labelled_scatter

X_fruits_normalized = StandardScaler().fit(X_fruits).transform(X_fruits)
mds = MDS(n_components=2)
X_fruits_mds = mds.fit_transform(X_fruits_normalized)
plot_labelled_scatter = (X_fruits_mds, y_fruits, ['Apple', 'mandarin', 'orange', 'lemon'])
plt.xlabel('first MDS feature')
plt.ylabel('second MDS feature')
plt.title('Fruit sample dataset')
