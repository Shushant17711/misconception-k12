"""TF-IDF + LR on gold transcripts, blended with the prior (task 4)."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from .data import load, labels
from .folds import folds
from .features import build_text
from .metric import map3_probs
from .log import record, load_probs

L = labels(); idx = {c: i for i, c in enumerate(L)}
tr = load('train'); y = tr.root_code.map(idx).values; f = folds().loc[tr.trace_id].values
X = [build_text(r, r.transcription_steps) for r in tr.itertuples()]
oof = np.full((len(tr), len(L)), 1e-6)
for k in range(5):
    m = f == k
    v = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000)
    Xtr = v.fit_transform([X[i] for i in np.where(~m)[0]]); Xva = v.transform([X[i] for i in np.where(m)[0]])
    clf = LogisticRegression(C=10, max_iter=2000).fit(Xtr, y[~m])
    oof[np.ix_(m, clf.classes_)] = clf.predict_proba(Xva)
print('tfidf alone', round(map3_probs(oof, y), 4))
prior = load_probs('prior')[0]
best = max((map3_probs(w * np.log(oof) + (1 - w) * np.log(prior), y), w) for w in np.linspace(0, 1, 11))
print('blend with prior (gold text) best', best)
