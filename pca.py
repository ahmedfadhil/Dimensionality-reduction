# PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from adspy_shared_utilities import plot_labelled_scatter

cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y=True)
X_normalized = StandardScaler().fit(X_cancer).transform(X_cancer)

pca = PCA(n_components=2).fit(X_normalized)
x_pca = pca.transform(X_normalized)
print(X_cancer.shape, x_pca.shape)
plot_labelled_scatter(X_pca,y_cancer,['malignant','benign'])

plt.xlable('First principle component')
plt.ylable('Second principle component')
plt.title('Breast cancer dataset')


