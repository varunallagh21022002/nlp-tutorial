from common import *
DATA = Path(__file__).resolve().parents[1] / "data"
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
