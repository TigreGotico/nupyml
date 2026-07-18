# nupyml benchmark scoreboard

Best score per task. Regenerate with `python bench/harness.py`.

| Task | Metric | Best submission | Best score |
|------|--------|-----------------|-----------:|
| [anomaly_detection](anomaly_detection/SCOREBOARD.md) | `roc_auc` | `baseline_isolation_forest` | 1.0000 |
| [blobs_clustering](blobs_clustering/SCOREBOARD.md) | `adjusted_rand_index` | `baseline_agglomerative` | 1.0000 |
| [density_estimation](density_estimation/SCOREBOARD.md) | `mean_log_likelihood` | `baseline_gmm` | -5.3229 |
| [diabetes_regression](diabetes_regression/SCOREBOARD.md) | `r2` | `baseline_ridge` | 0.3652 |
| [digits_classification](digits_classification/SCOREBOARD.md) | `accuracy` | `baseline_svc` | 0.9833 |
| [fraud_imbalanced](fraud_imbalanced/SCOREBOARD.md) | `macro_f1` | `baseline_gradient_boosting` | 0.7645 |
| [image_feature_classification](image_feature_classification/SCOREBOARD.md) | `accuracy` | `baseline_pixels_rf` | 0.9759 |
| [learning_to_rank](learning_to_rank/SCOREBOARD.md) | `ndcg@10` | `baseline_pointwise_ridge` | 0.9884 |
| [matrix_completion](matrix_completion/SCOREBOARD.md) | `rmse` | `baseline_matrix_factorization` | 0.8394 |
| [moons_classification](moons_classification/SCOREBOARD.md) | `accuracy` | `baseline_mlp` | 0.9700 |
| [multi_output_regression](multi_output_regression/SCOREBOARD.md) | `avg_r2` | `baseline_ridge` | 0.8174 |
| [multilabel_emotions](multilabel_emotions/SCOREBOARD.md) | `macro_f1_multilabel` | `baseline_binary_relevance` | 0.8106 |
| [ratings_recommender](ratings_recommender/SCOREBOARD.md) | `rmse` | `baseline_matrix_factorization` | 0.5572 |
| [semi_supervised](semi_supervised/SCOREBOARD.md) | `accuracy` | `baseline_self_training` | 0.9533 |
| [sine_forecast](sine_forecast/SCOREBOARD.md) | `rmse` | `baseline_holt_winters` | 1.1988 |
| [survival_risk](survival_risk/SCOREBOARD.md) | `concordance_index` | `baseline_coxph` | 0.8100 |
| [text_classification](text_classification/SCOREBOARD.md) | `accuracy` | `baseline_tfidf_logreg` | 0.8667 |
| [ts_classification](ts_classification/SCOREBOARD.md) | `accuracy` | `baseline_flatten_svc` | 0.9633 |
