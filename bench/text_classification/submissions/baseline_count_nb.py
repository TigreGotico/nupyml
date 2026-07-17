"""Bag-of-words counts + multinomial naive Bayes -- the classic text baseline."""
from nupyml.feature_extraction import CountVectorizer
from nupyml.naive_bayes import MultinomialNB


def solve(X_train, y_train, X_test):
    vec = CountVectorizer()
    Xtr = vec.fit_transform([str(d) for d in X_train])
    Xte = vec.transform([str(d) for d in X_test])
    return MultinomialNB().fit(Xtr.toarray(), y_train).predict(Xte.toarray())
