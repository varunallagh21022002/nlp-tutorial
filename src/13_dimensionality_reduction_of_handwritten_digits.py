from common import *
DATA = Path(__file__).resolve().parents[1] / "data"
dataset = load_digits()
X, y = dataset.data / 16.0, dataset.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                    stratify=y, random_state=SEED)
pca95 = PCA(n_components=0.95, svd_solver='full').fit(Xtr)
compressed = pca95.transform(Xte)
reconstructed = pca95.inverse_transform(compressed)
pca2 = PCA(n_components=2, svd_solver='full').fit(Xtr)
points = pca2.transform(Xte)
metrics = {'components_for_95pct': pca95.n_components_,
    'retained_variance': pca95.explained_variance_ratio_.sum(),
    'two_dimensional_variance': pca2.explained_variance_ratio_.sum(),
    'test_reconstruction_MSE': mean_squared_error(Xte, reconstructed)}
print(f'Train: {len(Xtr)} | Test: {len(Xte)} | Original features: 64')
for key, value in metrics.items(): print(f'{key}: {value:.4f}')
plt.scatter(points[:, 0], points[:, 1], c=yte, cmap='tab10', s=15, alpha=0.8)
plt.colorbar(ticks=range(10), label='True digit (visualization only)')
plt.xlabel('Principal component 1'); plt.ylabel('Principal component 2')
plt.title('Held-out digits projected into two dimensions')
plot_save('13_digits_pca')
