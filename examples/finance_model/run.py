"""Finance model compliance example — RBI ML Model Risk."""

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

import valiron

X, y = make_classification(n_samples=500, n_features=20, random_state=1)
X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

model = LogisticRegression(max_iter=200, random_state=1)
model.fit(X_train, y_train)

result = valiron.evaluate(
    model=model,
    X_test=X_test,
    y_test=y_test,
    regulation="rbi_ml_risk",
    use_case="credit_scoring",
)

print(f"Compliant: {result.compliant}")
print(f"Score:     {result.score:.1%}")
valiron.report(result, format="html", output="finance_compliance.html")
print("Report saved to finance_compliance.html")
