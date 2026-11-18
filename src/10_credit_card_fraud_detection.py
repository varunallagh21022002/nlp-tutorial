from common import *
DATA = Path(__file__).resolve().parents[1] / "data"
normal = rng.normal(0, 1, size=(1940, 5))
anomalies = rng.normal(2.8, 1.5, size=(60, 5))
X = pd.DataFrame(np.vstack([normal, anomalies]), columns=[
    'log_amount_z', 'distance_z', 'velocity_z', 'hour_risk_z', 'device_risk_z'])
y = np.r_[np.zeros(len(normal), dtype=int), np.ones(len(anomalies), dtype=int)]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                    stratify=y, random_state=SEED)
model = IsolationForest(n_estimators=200, contamination=0.03,
                        random_state=SEED, n_jobs=-1)
model.fit(Xtr)  # No labels supplied to the unsupervised estimator.
pred = (model.predict(Xte) == -1).astype(int)
risk = -model.decision_function(Xte)
print(f'Train: {len(Xtr)} | Test: {len(Xte)} | Test anomalies: {sum(yte)}')
print('Labels: 0 = normal; 1 = anomaly')
print(classification_report(yte, pred, digits=3, zero_division=0))
metrics = {'average_precision': average_precision_score(yte, risk),
           'anomaly_f1': f1_score(yte, pred),
           'prevalence_baseline_AP': float(yte.mean())}
for key, value in metrics.items(): print(f'{key}: {value:.4f}')
ConfusionMatrixDisplay.from_predictions(yte, pred, cmap='Blues', colorbar=False)
plt.title('Held-out anomaly detection')
plot_save('10_fraud_detection')
