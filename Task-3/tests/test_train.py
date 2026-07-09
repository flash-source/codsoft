def test_all_three_models_trained(trained_models):
    assert set(trained_models.keys()) == {
        "logistic_regression", "random_forest", "gradient_boosting",
    }


def test_predict_proba_in_valid_range(trained_models, processed_data):
    _, X_test, _, _, _ = processed_data
    for name, model in trained_models.items():
        proba = model.predict_proba(X_test)[:, 1]
        assert proba.min() >= 0.0 and proba.max() <= 1.0, f"{name} produced out-of-range probabilities"


def test_predictions_are_not_constant(trained_models, processed_data):
    # A model that collapsed to predicting one class for everyone would still
    # "run" without error -- this is the check that would actually catch that.
    _, X_test, _, _, _ = processed_data
    for name, model in trained_models.items():
        preds = set(model.predict(X_test))
        assert len(preds) > 1, f"{name} predicted a single class for every test row"


def test_class_weighting_applied_not_ignored(trained_models, processed_data):
    # With imbalance handled, recall on the minority class shouldn't collapse to ~0
    # (which is what you'd see if class_weight/sample_weight silently had no effect).
    from sklearn.metrics import recall_score
    _, X_test, _, y_test, _ = processed_data
    for name, model in trained_models.items():
        preds = model.predict(X_test)
        recall = recall_score(y_test, preds)
        assert recall > 0.3, f"{name} recall={recall:.3f} -- suspiciously low, check class weighting"
