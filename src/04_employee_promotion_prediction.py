from common import *
DATA = Path(__file__).resolve().parents[1] / "data"
n = 900
X = pd.DataFrame({'experience': rng.integers(0, 26, n),
    'performance_rating': rng.integers(1, 6, n),
    'training_score': rng.uniform(30, 100, n),
    'previous_achievements': rng.integers(0, 5, n)})
score = (0.7*(X.performance_rating-3) + 0.06*(X.training_score-65)
         + 0.35*X.previous_achievements + 0.05*X.experience - 1.8)
y = rng.binomial(1, 1/(1 + np.exp(-score)))
print('Labels: 0 = not promoted; 1 = promoted')
model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=15,
                               random_state=SEED)
metrics = evaluate(model, X, y, '04_employee_promotion')
print('Feature importances:')
print(pd.Series(model.feature_importances_, index=X.columns).round(3))
