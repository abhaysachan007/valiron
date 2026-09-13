"""HR model compliance example — EU AI Act (employment/recruitment)."""

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

import valiron

X, y = make_classification(n_samples=400, n_features=12, random_state=2)
X_train, X_test = X[:300], X[300:]
y_train, y_test = y[:300], y[300:]

model = DecisionTreeClassifier(max_depth=5, random_state=2)
model.fit(X_train, y_train)

result = valiron.evaluate(
    model=model,
    X_test=X_test,
    y_test=y_test,
    regulation="eu_ai_act",
    use_case="recruitment_screening",
    sensitive_features=["gender", "age_group"],
)

print(f"Compliant: {result.compliant}")
print(f"Score:     {result.score:.1%}")
valiron.report(result, format="html", output="hr_compliance.html")
