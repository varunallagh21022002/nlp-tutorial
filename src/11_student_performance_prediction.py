from common import *
DATA = Path(__file__).resolve().parents[1] / "data"
n = 900
X = pd.DataFrame({'attendance': rng.uniform(40, 100, n),
    'study_hours': rng.uniform(0, 30, n),
    'assignment_marks': rng.uniform(20, 100, n),
    'previous_scores': rng.uniform(20, 100, n),
    'participation': rng.integers(0, 11, n)})
y = (0.2*X.attendance + 0.6*X.study_hours + 0.25*X.assignment_marks
     + 0.3*X.previous_scores + 0.8*X.participation
     + rng.normal(0, 5, n)).clip(0, 100)
model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05,
                                  max_depth=2, random_state=SEED)
metrics = evaluate(model, X, y, '11_student_performance', classification=False)
