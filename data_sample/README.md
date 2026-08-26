# Data Sample

This directory provides small public samples derived from the text-feature
dataset and the BFS graph files.

## Files

- `aigc_new_with_style_features_sample.csv`: 120-row sample with the same 21
  columns as the full dataset.
- `data_schema.csv`: column names, data types in the full dataset, and brief
  field notes.
- `fans_bfs_nodes3_sample.csv`: 347 sampled graph nodes with the same 14
  columns as `fans_bfs_nodes3.csv`.
- `fans_bfs_edges3_sample.csv`: 300 sampled graph edges with the same 6
  columns as `fans_bfs_edges3.csv`.
- `graph_schema.csv`: field notes for the sampled graph node and edge files.

## Anonymization

The sample keeps the original column structure so that the released code can be
tested without the full dataset. Direct identifiers have been anonymized:

- `uid` values are replaced with `user_XXXX`.
- `wid` values are replaced with `post_XXXXX`.
- URLs, user mentions, and long numeric identifiers in `text_raw` are redacted.
- `create_time` keeps only the date and removes the exact posting time.
- Graph account identifiers in `seed_uid`, `parent_uid`, `uid`, `src_uid`, and
  `dst_uid` are replaced with stable `account_XXXXX` IDs shared across the node
  and edge samples.
- Graph display names and profile descriptions are redacted.
- Graph count fields are coarsened while remaining numeric.

The full text-feature dataset contains 39,681 rows. The full graph files contain
17,566 nodes and 19,121 edges. These samples are intended only for checking file
format and running lightweight smoke tests; they are not intended to reproduce
the paper's reported results.

## Note

The column name `laiyuan ` intentionally keeps the trailing space because the
original CSV uses that exact header.
