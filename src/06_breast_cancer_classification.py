from common import *
DATA = Path(__file__).resolve().parents[1] / "data"
dataset = load_breast_cancer(as_frame=True)
X, y = dataset.data, dataset.target
print('Classes:', dict(enumerate(dataset.target_names)))
model = make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf',
    gamma='scale', class_weight='balanced', random_state=SEED))
metrics = evaluate(model, X, y, '06_breast_cancer')
