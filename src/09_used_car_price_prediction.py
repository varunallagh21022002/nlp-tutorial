from common import *
DATA = Path(__file__).resolve().parents[1] / "data"
n = 1000
X = pd.DataFrame({'age': rng.uniform(0, 15, n),
    'kilometers_driven': rng.uniform(1000, 200000, n),
    'engine_capacity': rng.choice([1000, 1200, 1500, 2000, 2500], n),
    'fuel_type': rng.choice(['petrol', 'diesel', 'electric'], n),
    'transmission': rng.choice(['manual', 'automatic'], n)})
y = (2 + 0.006*X.engine_capacity*np.exp(-0.08*X.age)
     - 0.000012*X.kilometers_driven + 1.5*(X.transmission=='automatic')
     + 2*(X.fuel_type=='electric') + rng.normal(0, 0.6, n)).clip(0.5)
model = make_pipeline(tabular_preprocessor(X), RandomForestRegressor(
    n_estimators=150, min_samples_leaf=2, random_state=SEED, n_jobs=-1))
metrics = evaluate(model, X, y, '09_used_car_prices', classification=False)
