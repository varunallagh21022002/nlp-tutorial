from common import *
DATA = Path(__file__).resolve().parents[1] / "data"
df = pd.read_csv(DATA / 'sms.csv')
X, y = df.text.map(clean_text), df.label
assert not X.duplicated().any()
print('Messages:', len(df), '| Classes:', y.value_counts().to_dict())
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB())
metrics = evaluate(model, X, y, '07_sms_spam')
examples = ['Claim your free cash prize now', 'Meet me at the library tomorrow']
print('New SMS predictions:', model.predict([clean_text(s) for s in examples]))
