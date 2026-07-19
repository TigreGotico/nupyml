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
| [gp_regression](gp_regression/SCOREBOARD.md) | `r2` | `baseline_exact_gp` | 0.9951 |
| [hierarchical_forecast](hierarchical_forecast/SCOREBOARD.md) | `rmse` | `baseline_naive` | 0.4966 |
| [hyperbolic_hierarchy](hyperbolic_hierarchy/SCOREBOARD.md) | `spearman_rho` | `baseline_graph_distance` | 1.0000 |
| [image_denoising](image_denoising/SCOREBOARD.md) | `psnr_db` | `baseline_non_local_means` | 24.8594 |
| [image_feature_classification](image_feature_classification/SCOREBOARD.md) | `accuracy` | `baseline_pixels_rf` | 0.9759 |
| [image_segmentation](image_segmentation/SCOREBOARD.md) | `adjusted_rand` | `baseline_kmeans` | 1.0000 |
| [intermittent_forecast](intermittent_forecast/SCOREBOARD.md) | `rmse` | `baseline_mean_rate` | 1.9870 |
| [keypoint_matching](keypoint_matching/SCOREBOARD.md) | `accuracy` | `baseline_hog_lbp_rf` | 0.8815 |
| [learning_to_rank](learning_to_rank/SCOREBOARD.md) | `ndcg@10` | `baseline_pointwise_ridge` | 0.9884 |
| [link_prediction](link_prediction/SCOREBOARD.md) | `roc_auc` | `baseline_logistic` | 0.8576 |
| [matrix_completion](matrix_completion/SCOREBOARD.md) | `rmse` | `baseline_matrix_factorization` | 0.8394 |
| [moons_classification](moons_classification/SCOREBOARD.md) | `accuracy` | `baseline_mlp` | 0.9700 |
| [multi_output_regression](multi_output_regression/SCOREBOARD.md) | `avg_r2` | `baseline_ridge` | 0.8174 |
| [multilabel_emotions](multilabel_emotions/SCOREBOARD.md) | `macro_f1_multilabel` | `baseline_binary_relevance` | 0.8106 |
| [multivariate_forecast](multivariate_forecast/SCOREBOARD.md) | `rmse` | `baseline_var` | 0.3725 |
| [node_classification](node_classification/SCOREBOARD.md) | `accuracy` | `baseline_logistic` | 1.0000 |
| [ood_detection](ood_detection/SCOREBOARD.md) | `roc_auc` | `baseline_pca_recon` | 0.9695 |
| [point_cloud_classification](point_cloud_classification/SCOREBOARD.md) | `accuracy` | `baseline_bag_of_points` | 1.0000 |
| [ratings_recommender](ratings_recommender/SCOREBOARD.md) | `rmse` | `baseline_matrix_factorization` | 0.5572 |
| [regime_detection](regime_detection/SCOREBOARD.md) | `adjusted_rand` | `baseline_kmeans` | 0.7099 |
| [semi_supervised](semi_supervised/SCOREBOARD.md) | `accuracy` | `baseline_self_training` | 0.9533 |
| [sequence_mining](sequence_mining/SCOREBOARD.md) | `accuracy` | `baseline_bag_of_symbols` | 1.0000 |
| [sine_forecast](sine_forecast/SCOREBOARD.md) | `rmse` | `baseline_holt_winters` | 1.1988 |
| [survival_risk](survival_risk/SCOREBOARD.md) | `concordance_index` | `baseline_coxph` | 0.8100 |
| [symbolic_regression](symbolic_regression/SCOREBOARD.md) | `r2` | `baseline_poly_ridge` | 0.9997 |
| [text_classification](text_classification/SCOREBOARD.md) | `accuracy` | `baseline_tfidf_logreg` | 0.8667 |
| [timeseries_anomaly](timeseries_anomaly/SCOREBOARD.md) | `roc_auc` | `baseline_isolation_forest` | 0.9295 |
| [ts_classification](ts_classification/SCOREBOARD.md) | `accuracy` | `baseline_flatten_svc` | 0.9633 |
| [volatility_forecast](volatility_forecast/SCOREBOARD.md) | `rmse` | `baseline_garch` | 2.6227 |
