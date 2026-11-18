from common import *
DATA = Path(__file__).resolve().parents[1] / "data"
dataset = load_iris(as_frame=True)
X, y = dataset.data, dataset.target
print('Classes:', dict(enumerate(dataset.target_names)))
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
metrics = evaluate(model, X, y, '03_iris_knn')
