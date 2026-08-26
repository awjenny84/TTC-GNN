# Code

This directory contains the essential scripts used for preprocessing, training,
evaluation, ablation studies, significance testing, and result analysis.

The files are kept in a flat layout so that the original local imports continue
to work without modifying the experiment scripts.

## Preprocessing

- `create_engagement_labels.py`: create engagement labels.
- `extract_style_features.py`: extract text style and sentiment features.
- `build_node_features_raw.py`: build graph node features and edge index files.
- `tail_cutoff_and_binning.py`: engagement tail cutoff and class binning.

## Model Components

- `topology_feature_encoder.py`: topology feature encoder used by graph and
  multimodal models.

## Training and Baselines

- `train_ttc_gnn_main.py`: main TTC-GNN multi-seed training script.
- `train_ttc_gnn_per_bot_811.py`: per-bot 8/1/1 split training script.
- `train_ttc_gnn_hparam.py`: configurable TTC-GNN training script for hyperparameter runs.
- `train_baseline_graph_only.py`: graph-only baseline.
- `train_text_baselines_bert_roberta.py`: text-only BERT/RoBERTa baseline.
- `role_conditioned_text_multiclass.py`: role-conditioned text baseline.
- `train_text_film_baseline.py`: text + FiLM baseline.

## Ablations and Variants

- `train_ablation_network_only.py`: network-only ablation.
- `train_ablation_persona_only.py`: persona-only ablation.
- `train_ablation_style_only.py`: style-only ablation.
- `train_ttc_gnn_gated.py`: gated fusion variant.
- `train_ttc_gnn_mask_concat.py`: mask-concat fusion variant.

## Evaluation and Analysis

- `grid_search_ttc_gnn.py`: grid search utility.
- `run_hyperparam_sweep_macro_f1.py`: hyperparameter sweep runner.
- `significance_test_temporal.py`: paired significance testing.
- `plot_val_macro_f1_vs_epoch_gh.py`: graph-hidden validation curve plotting.
- `plot_val_macro_f1_vs_epoch_ml.py`: max-length validation curve plotting.
- `shap_group_analysis.py`: SHAP feature analysis.
- `analy_dca_attention.py`: dual co-attention inspection utility.

## Shell Scripts

- `run_full_hparam_sweep.sh`: full hyperparameter sweep example.
- `run_gh_maxlen_sweep.sh`: graph-hidden and max-length sweep example.
