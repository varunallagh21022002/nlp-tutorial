from common import *
DATA = Path(__file__).resolve().parents[1] / "data"
n = 800
X = pd.DataFrame({'area': rng.uniform(500, 3500, n),
    'bedrooms': rng.integers(1, 6, n),
    'bathrooms': rng.integers(1, 5, n),
    'age': rng.integers(0, 41, n),
    'location': rng.choice(['central', 'suburb', 'outskirts'], n)})
premium = X.location.map({'central': 30, 'suburb': 15, 'outskirts': 0})
y = (0.035*X.area + 5*X.bedrooms + 4*X.bathrooms
     - 0.5*X.age + premium + rng.normal(0, 7, n))
model = make_pipeline(tabular_preprocessor(X), LinearRegression())
metrics = evaluate(model, X, y, '01_house_prices', classification=False)
