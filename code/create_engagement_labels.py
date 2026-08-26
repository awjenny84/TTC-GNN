import argparse
import json
import os

import numpy as np
import pandas as pd


def find_aigc_csv(directory):
    candidates = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        lower = name.lower()
        if lower.endswith('.csv') and 'aigc_new_with_style_features' in lower:
            # Prefer the original feature table rather than an already generated label table.
            generated = 'engagement' in lower or 'labeled' in lower or 'labelled' in lower
            candidates.append((generated, len(name), name, path))
    if not candidates:
        raise FileNotFoundError("No CSV containing 'aigc_new_with_style_features' was found.")
    return sorted(candidates)[0][3]


def normalize_id(series):
    return series.astype(str).str.strip().str.strip("'").str.strip('"')


def discrete_tertiles(values):
    """Return observed-value empirical tertiles using NumPy's higher method."""
    values = np.asarray(values, dtype=np.int64)
    if values.size == 0:
        raise ValueError('Chronological training split has no posts with comments_count > 0.')
    try:
        result = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0], method='higher')
    except TypeError:  # NumPy < 1.22
        result = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0], interpolation='higher')
    q1, q2 = int(result[0]), int(result[1])
    if q1 > q2:
        raise AssertionError('Invalid thresholds: q1 > q2')
    return q1, q2


def assign_labels(comment_counts, q1, q2):
    """Assign 0=low, 1=medium, 2=high using two fixed thresholds."""
    values = np.asarray(comment_counts, dtype=np.int64)
    labels = np.select(
        [
            values <= q1,
            (values > q1) & (values <= q2),
            values > q2,
        ],
        [0, 1, 2],
        default=0,
    )
    return labels.astype(np.int8)


def prepare_input(data, comment_column, time_column):
    required = ['uid', 'wid', comment_column]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError('Input CSV is missing columns: {}'.format(missing))

    # Backward compatibility with the typo present in some old files.
    if time_column not in data.columns:
        if time_column == 'create_time' and 'create_tiime' in data.columns:
            data = data.rename(columns={'create_tiime': 'create_time'})
        else:
            raise ValueError("Input CSV is missing time column '{}'.".format(time_column))

    data = data.copy()
    data['uid'] = normalize_id(data['uid'])
    data['wid'] = normalize_id(data['wid'])

    empty_uid = data['uid'].isin(['', 'nan', 'None'])
    empty_wid = data['wid'].isin(['', 'nan', 'None'])
    if empty_uid.any() or empty_wid.any():
        raise ValueError(
            'Found empty uid/wid rows: empty_uid={} empty_wid={}.'.format(
                int(empty_uid.sum()), int(empty_wid.sum())
            )
        )

    comments = pd.to_numeric(data[comment_column], errors='coerce')
    invalid = comments.isna() | (comments < 0) | (comments % 1 != 0)
    if invalid.any():
        examples = data.loc[invalid, comment_column].head(5).tolist()
        raise ValueError(
            '{} must contain non-negative integers; invalid rows={}, examples={}'.format(
                comment_column, int(invalid.sum()), examples
            )
        )
    data['comments_count'] = comments.astype(np.int64)

    parsed_time = pd.to_datetime(data[time_column], errors='coerce')
    bad_time = parsed_time.isna()
    if bad_time.any():
        examples = data.loc[bad_time, ['uid', 'wid', time_column]].head(5).to_dict('records')
        raise ValueError(
            "Invalid or missing timestamps in '{}': rows={}, examples={}".format(
                time_column, int(bad_time.sum()), examples
            )
        )
    data['__parsed_time'] = parsed_time
    return data


def chronological_split(data):
    """Replicate the 811 trainer's per-bot chronological 80/10/10 split."""
    split_series = pd.Series(index=data.index, dtype='object')
    stats = []

    for uid, group in data.groupby('uid', sort=True):
        ordered = group.sort_values(['__parsed_time', 'wid'], kind='mergesort')
        n = len(ordered)
        if n < 3:
            raise ValueError(
                'Bot {} has only {} posts; at least 3 are required for an 8/1/1 split.'.format(uid, n)
            )

        n_train = int(np.floor(0.8 * n))
        n_val = int(np.floor(0.1 * n))
        n_test = n - n_train - n_val

        if n_val == 0:
            n_val = 1
            n_train -= 1
        if n_test == 0:
            n_test = 1
            n_train -= 1
        if n_train <= 0:
            raise ValueError('Bot {} does not have enough posts for non-empty 8/1/1 splits.'.format(uid))

        train_idx = ordered.index[:n_train]
        val_idx = ordered.index[n_train:n_train + n_val]
        test_idx = ordered.index[n_train + n_val:]

        split_series.loc[train_idx] = 'train'
        split_series.loc[val_idx] = 'val'
        split_series.loc[test_idx] = 'test'
        stats.append((uid, n, n_train, n_val, n_test))

    if split_series.isna().any():
        raise AssertionError('Some posts were not assigned to train/val/test.')

    return split_series, stats


def class_statistics(labels):
    counts = pd.Series(labels).value_counts().reindex(range(3), fill_value=0).astype(int)
    total = int(counts.sum())
    ratios = (counts / total).to_dict() if total else {0: 0.0, 1: 0.0, 2: 0.0}
    return counts.to_dict(), ratios


def default_output_path(input_path):
    root, ext = os.path.splitext(input_path)
    return root + '_with_engagement_class' + (ext or '.csv')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Create one engagement_class label for every post. Thresholds are estimated '
            'from the chronological training portion (earliest 80% of each bot) only, '
            'then applied to train/val/test posts.'
        )
    )
    parser.add_argument('--input', default='', help='Input feature CSV; default: auto-detect aigc_new_with_style_features*.csv.')
    parser.add_argument('--output', default='', help='Output CSV. Default: <input>_with_engagement_class.csv')
    parser.add_argument('--comment-column', default='comments_count')
    parser.add_argument('--time-column', default='create_time')
    parser.add_argument('--thresholds-file', default='', help='Optional JSON file for the derived q1/q2 and class statistics.')
    parser.add_argument(
        '--write-split-column',
        action='store_true',
        help="Also write a temporary 'chronological_split' column for inspection. The 811 trainer does not require it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(args.input) if args.input else find_aigc_csv(base_dir)
    output_path = os.path.abspath(args.output) if args.output else default_output_path(input_path)

    raw = pd.read_csv(input_path, encoding='utf-8-sig', dtype={'uid': str, 'wid': str})
    data = prepare_input(raw, args.comment_column, args.time_column)

    split_series, bot_stats = chronological_split(data)

    # IMPORTANT: derive thresholds only from chronological TRAIN posts with positive comments.
    train_positive = data.loc[
        (split_series == 'train') & (data['comments_count'] > 0),
        'comments_count',
    ]
    q1, q2 = discrete_tertiles(train_positive.to_numpy())

    labels = assign_labels(data['comments_count'].to_numpy(), q1, q2)

    # Preserve the complete original feature table and append engagement_class.
    output = data.drop(columns=['__parsed_time']).copy()
    output['engagement_class'] = labels
    if args.write_split_column:
        output['chronological_split'] = split_series.values

    # Verify that uid/wid order and row count are preserved.
    if len(output) != len(raw):
        raise AssertionError('Output row count differs from input row count.')
    if output['engagement_class'].isna().any():
        raise AssertionError('Some posts do not have engagement_class labels.')
    observed = set(output['engagement_class'].astype(int).unique().tolist())
    if not observed.issubset({0, 1, 2}):
        raise AssertionError('Unexpected engagement classes: {}'.format(sorted(observed)))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    output.to_csv(output_path, index=False, encoding='utf-8-sig')

    overall_counts, overall_ratios = class_statistics(output['engagement_class'])
    print('[INFO] Label definition: 0=low, 1=medium, 2=high')
    print('[INFO] Threshold source: chronological TRAIN posts only (earliest 80% per bot, positive comments only)')
    print('[INFO] thresholds: q1={} q2={}'.format(q1, q2))
    print('[INFO] class rule: low <= {}; medium {} < comments <= {}; high > {}'.format(q1, q1, q2, q2))
    print('[INFO] total bots={} posts={}'.format(output['uid'].nunique(), len(output)))
    print('[INFO] class counts={} ratios={}'.format(overall_counts, {k: round(v, 6) for k, v in overall_ratios.items()}))
    print('[INFO] per-bot chronological split examples (uid,total,train,val,test): {}'.format(bot_stats[:5]))
    print('[DONE] Every post now has engagement_class. Output: {}'.format(output_path))

    if args.thresholds_file:
        thresholds_path = os.path.abspath(args.thresholds_file)
        payload = {
            'split_method': 'per-bot chronological 80/10/10',
            'threshold_source': 'positive-comment posts in chronological train split only',
            'comment_column': args.comment_column,
            'time_column': args.time_column,
            'q1': int(q1),
            'q2': int(q2),
            'class_definition': {
                '0': 'comments_count <= q1',
                '1': 'q1 < comments_count <= q2',
                '2': 'comments_count > q2',
            },
            'num_bots': int(output['uid'].nunique()),
            'num_posts': int(len(output)),
            'class_counts': {str(k): int(v) for k, v in overall_counts.items()},
            'class_ratios': {str(k): float(v) for k, v in overall_ratios.items()},
        }
        os.makedirs(os.path.dirname(thresholds_path) or '.', exist_ok=True)
        with open(thresholds_path, 'w', encoding='utf-8') as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        print('[INFO] Threshold metadata: {}'.format(thresholds_path))


if __name__ == '__main__':
    main()