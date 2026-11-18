from common import *
DATA = Path(__file__).resolve().parents[1] / "data"
n = 1000
X = pd.DataFrame({'income': rng.uniform(3, 30, n),
    'credit_history': rng.binomial(1, 0.75, n),
    'loan_amount': rng.uniform(2, 60, n),
    'employment': rng.choice(['salaried', 'self_employed', 'unemployed'], n)})
job = X.employment.map({'salaried': 0.8, 'self_employed': 0.2, 'unemployed': -1})
score = 0.13*X.income + 2*X.credit_history - 0.07*X.loan_amount + job - 1.5
y = rng.binomial(1, 1/(1 + np.exp(-score)))
print('Labels: 0 = rejected; 1 = approved')
model = make_pipeline(tabular_preprocessor(X),
                      LogisticRegression(max_iter=2000, random_state=SEED))
metrics = evaluate(model, X, y, '02_loan_approval')
