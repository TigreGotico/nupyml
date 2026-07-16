"""Train an MLP on two moons and print train/test accuracy."""
from nupyml.datasets import make_moons
from nupyml.model_selection import train_test_split
from nupyml.nn import MLPClassifier
from nupyml.linear_model import LogisticRegression

X, y = make_moons(500, noise=0.2, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

mlp = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=300, random_state=0)
mlp.fit(Xtr, ytr)
lin = LogisticRegression().fit(Xtr, ytr)

print(f"logistic regression: {lin.score(Xte, yte):.3f}")
print(f"mlp (64,64):         {mlp.score(Xte, yte):.3f}")
