"""Append-only CSV logger for experiment results with file locking."""
import csv
import fcntl
import os
import time
import uuid

RESULTS_PATH = 'experiments/results/results.csv'
COLUMNS = ['experiment_id', 'model', 'seed', 'eval_mode', 'metric',
           'value', 'epoch', 'wall_time', 'config_path']


def log_result(model, seed, eval_mode, metric, value, epoch,
               wall_time=None, config_path='', results_path=None):
    """Append one row to the results CSV.

    Args:
        model: model name (e.g. 'tgn', 'tgat')
        seed: random seed used
        eval_mode: 'legacy' or 'tgb'
        metric: metric name (e.g. 'ap', 'auc', 'mrr')
        value: metric value (float)
        epoch: training epoch
        wall_time: wall-clock time in seconds (defaults to current time)
        config_path: path to config YAML used
        results_path: override default CSV path
    """
    path = results_path or RESULTS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if wall_time is None:
        wall_time = time.time()
    row = {
        'experiment_id': str(uuid.uuid4())[:8],
        'model': model,
        'seed': seed,
        'eval_mode': eval_mode,
        'metric': metric,
        'value': value,
        'epoch': epoch,
        'wall_time': wall_time,
        'config_path': config_path,
    }
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)
