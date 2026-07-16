"""Quick tour: a handful of classic estimators on the same dataset."""
from nupyml.datasets import make_classification
from nupyml.model_selection import train_test_split, cross_val_score
from nupyml.pipeline import make_pipeline
from nupyml.preprocessing import StandardScaler
from nupyml.linear_model import LogisticRegression
from nupyml.svm import SVC
from nupyml.tree import DecisionTreeClassifier
from nupyml.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from nupyml.neighbors import KNeighborsClassifier
from nupyml.naive_bayes import GaussianNB

X, y = make_classification(n_samples=600, n_features=12, n_informative=5,
                           random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression()),
    "svc-rbf": make_pipeline(StandardScaler(), SVC(random_state=0)),
    "tree": DecisionTreeClassifier(max_depth=6),
    "forest": RandomForestClassifier(n_estimators=100, random_state=0),
    "hist-gb": HistGradientBoostingClassifier(max_iter=100),
    "knn": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "gauss-nb": GaussianNB(),
}
for name, model in models.items():
    acc = model.fit(Xtr, ytr).score(Xte, yte)
    print(f"{name:10s} {acc:.3f}")
