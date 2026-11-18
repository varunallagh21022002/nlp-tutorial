from common import *
DATA = Path(__file__).resolve().parents[1] / "data"
n = 1200
X = pd.DataFrame({'usage_gb': rng.uniform(1, 100, n),
    'tenure_months': rng.integers(1, 73, n),
    'contract': rng.choice(['monthly', 'annual', 'two_year'], n),
    'payment_method': rng.choice(['automatic', 'manual'], n),
    'monthly_charges': rng.uniform(20, 120, n)})
score = (1.4*(X.contract=='monthly') + 0.5*(X.payment_method=='manual')
         + 0.025*X.monthly_charges - 0.045*X.tenure_months
         + 0.005*X.usage_gb - 2)
y = rng.binomial(1, 1/(1 + np.exp(-score)))
print('Labels: 0 = retained; 1 = churned')
model = make_pipeline(tabular_preprocessor(X), RandomForestClassifier(
    n_estimators=150, max_depth=8, min_samples_leaf=5,
    class_weight='balanced', random_state=SEED, n_jobs=-1))
metrics = evaluate(model, X, y, '05_customer_churn')
