"""Assignment definitions. Data are generated locally unless a benchmark is named."""
PRACTICALS = []
def add(number, title, model, aim, data, theory, code, limitation):
    PRACTICALS.append(dict(number=number, title=title, model=model, aim=aim,
        data=data, theory=theory, code=code.strip(), limitation=limitation))

add(1, 'House Price Prediction', 'Linear Regression',
'Predict house prices using area, bedrooms, bathrooms, age and location features.',
'800 generated houses. Area: square feet; age: years; price: INR lakh. Location has three categories.',
'Linear regression estimates coefficients by minimizing squared residuals. Categorical locations are one-hot encoded; numerical features are imputed and scaled using training data only.',
'''
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
''', 'Generated prices follow a near-linear rule; the score does not establish performance on real property transactions.')

add(2, 'Loan Approval Prediction', 'Logistic Regression',
'Predict approval from income, credit history, loan amount and employment information.',
'1,000 generated applications. Income and loan amount are in INR lakh; credit history is a binary indicator.',
'Logistic regression maps a linear score to a probability using the sigmoid function. A stratified split preserves label proportions; scaling and employment encoding are fitted only on training rows.',
'''
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
''', 'The labels are simulated for teaching. Real lending requires representative data, policy checks and fairness evaluation.')

add(3, 'Iris Flower Classification', 'K-Nearest Neighbors (KNN)',
'Classify Setosa, Versicolor and Virginica from sepal and petal measurements.',
'scikit-learn Iris benchmark: 150 flowers, four measurements in centimetres and three species.',
'KNN predicts the majority class of the five closest training observations. Standardization prevents a feature with a larger numerical scale from dominating Euclidean distance.',
'''
dataset = load_iris(as_frame=True)
X, y = dataset.data, dataset.target
print('Classes:', dict(enumerate(dataset.target_names)))
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
metrics = evaluate(model, X, y, '03_iris_knn')
''', 'The test split contains only 38 flowers. Results depend on the split and do not measure performance on every flower species.')

add(4, 'Employee Promotion Prediction', 'Decision Tree',
'Predict promotion using experience, performance rating, training score and prior achievements.',
'900 generated employee records with simulated promotion labels.',
'A decision tree recursively partitions the feature space using impurity reduction. Limiting tree depth and minimum leaf size reduces overfitting.',
'''
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
''', 'Simulated labels encode a teaching rule; feature importance is not evidence that a factor causes promotion.')

add(5, 'Customer Churn Prediction', 'Random Forest',
'Identify customers likely to leave a telecom service using usage, contracts, payment method and monthly charges.',
'1,200 generated telecom customers; usage in GB, tenure in months, and monthly charges in currency units.',
'A random forest averages multiple trees trained on bootstrap samples with randomized feature selection. Class weighting accounts for unequal label frequencies.',
'''
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
''', 'The generated churn probabilities are noisy. Recall for the churn class matters more than accuracy alone for retention planning.')

add(6, 'Breast Cancer Classification', 'Support Vector Machine (SVM)',
'Classify diagnostic feature vectors as malignant or benign.',
'scikit-learn Wisconsin Diagnostic Breast Cancer benchmark: 569 samples and 30 numerical features.',
'An RBF-kernel SVM learns a nonlinear decision boundary with a maximum-margin objective. Standardization is fitted on training data to avoid test-set leakage.',
'''
dataset = load_breast_cancer(as_frame=True)
X, y = dataset.data, dataset.target
print('Classes:', dict(enumerate(dataset.target_names)))
model = make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf',
    gamma='scale', class_weight='balanced', random_state=SEED))
metrics = evaluate(model, X, y, '06_breast_cancer')
''', 'This is a benchmark exercise, not a validated clinical diagnostic system. Review malignant-class recall as well as accuracy.')

add(7, 'Spam SMS Detection', 'TF-IDF + Naive Bayes',
'Clean SMS text, extract TF-IDF features and classify spam versus legitimate messages.',
'80 original, manually authored teaching messages: 40 spam and 40 ham. No downloaded SMS corpus or personal messages are used.',
'Text is lowercased, URLs and numbers are normalized, and punctuation is removed. TF-IDF downweights frequent terms; Multinomial Naive Bayes uses class-conditional token counts with additive smoothing.',
'''
df = pd.read_csv(DATA / 'sms.csv')
X, y = df.text.map(clean_text), df.label
assert not X.duplicated().any()
print('Messages:', len(df), '| Classes:', y.value_counts().to_dict())
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB())
metrics = evaluate(model, X, y, '07_sms_spam')
examples = ['Claim your free cash prize now', 'Meet me at the library tomorrow']
print('New SMS predictions:', model.predict([clean_text(s) for s in examples]))
''', 'The small authored corpus illustrates the pipeline only. Its test metrics are not a real-world spam benchmark.')

add(8, 'Customer Segmentation', 'K-Means Clustering',
'Group customers by age, annual income and spending score without predefined labels.',
'450 generated customers from three overlapping profiles. Income is in thousands of currency units; spending score ranges from 1 to 100.',
'K-Means minimizes within-cluster squared distance. Features are standardized, candidate values of k are compared by silhouette score, and centres are converted back to original units for interpretation.',
'''
centres = [[25, 35, 75], [45, 90, 65], [55, 50, 25]]
parts = [rng.normal(c, [8, 15, 14], size=(150, 3)) for c in centres]
X = pd.DataFrame(np.vstack(parts), columns=['age', 'annual_income', 'spending_score'])
X['age'] = X.age.clip(18, 80)
X['annual_income'] = X.annual_income.clip(10, 160)
X['spending_score'] = X.spending_score.clip(1, 100)
scaler = StandardScaler()
Z = scaler.fit_transform(X)
candidates = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
    candidates[k] = (silhouette_score(Z, km.labels_), km)
best_k = max(candidates, key=lambda k: candidates[k][0])
score, model = candidates[best_k]
print('Silhouette by k:', {k: round(v[0], 3) for k, v in candidates.items()})
print('Selected k:', best_k)
print('Cluster sizes:', np.bincount(model.labels_))
print('Centres in original units:')
print(pd.DataFrame(scaler.inverse_transform(model.cluster_centers_),
                   columns=X.columns).round(2).to_string(index=False))
plt.scatter(X.annual_income, X.spending_score, c=model.labels_, cmap='viridis', s=18)
plt.xlabel('Annual income (thousands)'); plt.ylabel('Spending score')
plt.title('Customer segments; age also used for clustering')
plot_save('08_customer_segments')
metrics = {'selected_k': best_k, 'silhouette': score}
''', 'Silhouette is an internal measure computed on the same data used for clustering. Cluster numbers are arbitrary, not externally verified customer categories.')

add(9, 'Used Car Price Prediction', 'Random Forest Regressor',
'Predict used-car prices from age, kilometres driven, engine capacity, fuel type and transmission.',
'1,000 generated cars. Price: INR lakh; distance: kilometres; engine: cubic centimetres.',
'Random forest regression averages decision-tree predictions and can represent nonlinear relationships. Fuel and transmission are encoded inside the training pipeline.',
'''
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
''', 'The generated price rule omits brand, condition, accident history and market changes; benchmark accuracy cannot establish resale valuation accuracy.')

add(10, 'Credit Card Fraud Detection', 'Isolation Forest',
'Detect unusual transactions from numerical features using an unsupervised anomaly detector.',
'2,000 generated transactions: 1,940 normal and 60 anomalies. Labels are used only for stratification and final evaluation, not model fitting.',
'Isolation Forest isolates observations using random partitions; anomalies tend to require shorter paths. The contamination rate is fixed in advance at 3 percent, not tuned on held-out labels.',
'''
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
''', 'An anomaly is not proof of fraud. The generated anomalies are deliberately separated, making this easier than most real transaction datasets.')

add(11, 'Student Performance Prediction', 'Gradient Boosting Regressor',
'Predict a final academic score from attendance, study hours, assignments, previous scores and participation.',
'900 generated students. Marks and attendance are percentages; study hours are weekly. The regression option allowed by the assignment is used.',
'Gradient boosting fits successive shallow trees to reduce the residual error of an additive model. Learning rate and tree complexity control model capacity.',
'''
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
''', 'Predictive associations in synthetic records are not causal explanations of student achievement.')

add(12, 'Movie Review Sentiment Analysis', 'CountVectorizer + SVM',
'Tokenize reviews, remove stop words, vectorize counts and classify positive or negative sentiment.',
'80 original short teaching reviews, balanced between positive and negative. The corpus is intentionally small and stored locally.',
'CountVectorizer performs word tokenization and stop-word removal. Negation words are retained; unigrams and bigrams feed a linear SVM so phrases such as "not good" remain distinguishable.',
'''
df = pd.read_csv(DATA / 'reviews.csv')
X, y = df.text.map(clean_text), df.label
assert not X.duplicated().any()
stop_words = sorted(set(ENGLISH_STOP_WORDS) - {'not', 'no', 'nor', 'never'})
vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=(1, 2))
print('Example tokens:', vectorizer.build_analyzer()(clean_text(X.iloc[0])))
model = make_pipeline(vectorizer, LinearSVC(C=1.0, random_state=SEED))
metrics = evaluate(model, X, y, '12_movie_sentiment')
examples = ['A wonderful story and excellent acting',
            'A boring story with terrible acting']
print('New review predictions:', model.predict([clean_text(s) for s in examples]))
''', 'Accuracy is below the 0.50 majority baseline on this split. The small corpus is insufficient for reliable sentiment learning; use a larger independent review dataset before drawing performance conclusions.')

add(13, 'Dimensionality Reduction of Handwritten Digits', 'Principal Component Analysis (PCA)',
'Reduce handwritten-digit image features while retaining information and visualize samples in two dimensions.',
'scikit-learn Digits benchmark: 1,797 images, each 8 x 8 pixels with intensities from 0 to 16.',
'PCA finds orthogonal directions of maximum variance. A 95-percent-variance representation is compared with a separate two-component visualization. PCA is fitted only on training images; labels are used only to colour the held-out plot.',
'''
dataset = load_digits()
X, y = dataset.data / 16.0, dataset.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                    stratify=y, random_state=SEED)
pca95 = PCA(n_components=0.95, svd_solver='full').fit(Xtr)
compressed = pca95.transform(Xte)
reconstructed = pca95.inverse_transform(compressed)
pca2 = PCA(n_components=2, svd_solver='full').fit(Xtr)
points = pca2.transform(Xte)
metrics = {'components_for_95pct': pca95.n_components_,
    'retained_variance': pca95.explained_variance_ratio_.sum(),
    'two_dimensional_variance': pca2.explained_variance_ratio_.sum(),
    'test_reconstruction_MSE': mean_squared_error(Xte, reconstructed)}
print(f'Train: {len(Xtr)} | Test: {len(Xte)} | Original features: 64')
for key, value in metrics.items(): print(f'{key}: {value:.4f}')
plt.scatter(points[:, 0], points[:, 1], c=yte, cmap='tab10', s=15, alpha=0.8)
plt.colorbar(ticks=range(10), label='True digit (visualization only)')
plt.xlabel('Principal component 1'); plt.ylabel('Principal component 2')
plt.title('Held-out digits projected into two dimensions')
plot_save('13_digits_pca')
''', 'Two principal components retain much less information than the 95-percent representation; class overlap in two dimensions is expected.')

add(14, 'News Category Prediction', 'TF-IDF + Multinomial Logistic Regression',
'Classify headlines into Sports, Business, Technology, Politics and Entertainment.',
'100 original teaching headlines: 20 per category, with no downloaded news text. Each headline is unique.',
'TF-IDF represents unigram and bigram evidence. Logistic regression with the lbfgs solver optimizes multinomial softmax loss for the five-class task.',
'''
df = pd.read_csv(DATA / 'news.csv')
X, y = df.text.map(clean_text), df.label
assert not X.duplicated().any()
print('Headlines:', len(df), '| Categories:', sorted(y.unique()))
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)),
    LogisticRegression(solver='lbfgs', max_iter=2000, random_state=SEED))
metrics = evaluate(model, X, y, '14_news_categories')
examples = ['Team wins the football tournament final',
            'Software company releases a new artificial intelligence chip']
print('New headline predictions:', model.predict([clean_text(s) for s in examples]))
''', 'Topic keywords are comparatively clear in this teaching corpus; production news can span multiple categories or change vocabulary.')

add(15, 'Sales Prediction', 'AdaBoost Regressor',
'Predict future sales from advertising, previous sales, product price, discounts and seasonal features.',
'600 generated daily observations; sales are units, advertising and price are currency units, discount is a fraction. The final 150 days form the test period.',
'AdaBoost combines weak regression trees while increasing attention to poorly predicted observations. Evaluation uses a chronological split; previous-day sales are observable for rolling one-day-ahead prediction.',
'''
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
''', 'This is rolling one-day-ahead forecasting with known advertising and price. Multi-day forecasting would need predicted lags and a separate evaluation.')
