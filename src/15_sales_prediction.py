from common import *
DATA = Path(__file__).resolve().parents[1] / "data"
n = 601
advertising = rng.uniform(10, 100, n)
price = rng.uniform(15, 40, n)
discount = rng.uniform(0, 0.3, n)
season = np.sin(2*np.pi*np.arange(n)/30)
sales = np.zeros(n); sales[0] = 100
for t in range(1, n):
    sales[t] = max(0, 30 + 0.45*sales[t-1] + 1.2*advertising[t]
        - 1.3*price[t] + 80*discount[t] + 15*season[t]
        + rng.normal(0, 8))
X = pd.DataFrame({'advertising': advertising[1:],
    'previous_sales': sales[:-1], 'price': price[1:],
    'discount': discount[1:], 'seasonal_signal': season[1:]})
y = sales[1:]
model = AdaBoostRegressor(n_estimators=150, learning_rate=0.05,
                          loss='square', random_state=SEED)
metrics = evaluate(model, X, y, '15_sales_prediction',
                   classification=False, chronological=True)
cut = int(len(y)*0.75)
metrics['persistence_MAE'] = mean_absolute_error(y[cut:], X.previous_sales.iloc[cut:])
print(f'Previous-day persistence MAE: {metrics["persistence_MAE"]:.4f}')
