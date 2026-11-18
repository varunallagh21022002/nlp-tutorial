from common import *
DATA = Path(__file__).resolve().parents[1] / "data"
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
