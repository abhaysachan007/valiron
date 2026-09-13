"""Healthcare model compliance example — CDSCO MDSW."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import valiron

X, y = make_classification(n_samples=500, n_features=15, random_state=0)
X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

model = RandomForestClassifier(n_estimators=50, random_state=0)
model.fit(X_train, y_train)

result = valiron.evaluate(
    model=model,
    X_test=X_test,
    y_test=y_test,
    regulation="cdsco_mdsw",
    use_case="diagnostic_aid",
)

print(f"Compliant: {result.compliant}")
print(f"Score:     {result.score:.1%}")
print(f"Failing:   {result.failing_checks or 'none'}")

valiron.report(result, format="html", output="healthcare_compliance.html")
print("Report saved to healthcare_compliance.html")
