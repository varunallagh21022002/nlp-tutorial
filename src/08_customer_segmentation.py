from common import *
DATA = Path(__file__).resolve().parents[1] / "data"
centres = [[25, 35, 75], [45, 90, 65], [55, 50, 25]]
parts = [rng.normal(c, [8, 15, 14], size=(150, 3)) for c in centres]
X = pd.DataFrame(np.vstack(parts), columns=['age', 'annual_income', 'spending_score'])
X['age'] = X.age.clip(18, 80)
X['annual_income'] = X.annual_income.clip(10, 160)
X['spending_score'] = X.spending_score.clip(1, 100)
scaler = StandardScaler()
Z = scaler.fit_transform(X)
candidates = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
    candidates[k] = (silhouette_score(Z, km.labels_), km)
best_k = max(candidates, key=lambda k: candidates[k][0])
score, model = candidates[best_k]
print('Silhouette by k:', {k: round(v[0], 3) for k, v in candidates.items()})
print('Selected k:', best_k)
print('Cluster sizes:', np.bincount(model.labels_))
print('Centres in original units:')
print(pd.DataFrame(scaler.inverse_transform(model.cluster_centers_),
                   columns=X.columns).round(2).to_string(index=False))
plt.scatter(X.annual_income, X.spending_score, c=model.labels_, cmap='viridis', s=18)
plt.xlabel('Annual income (thousands)'); plt.ylabel('Spending score')
plt.title('Customer segments; age also used for clustering')
plot_save('08_customer_segments')
metrics = {'selected_k': best_k, 'silhouette': score}
