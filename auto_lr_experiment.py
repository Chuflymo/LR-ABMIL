import argparse
import json
import os
import subprocess
from datetime import datetime

DEFAULT_CONFIG = {
    'base_cmd': 'CUDA_VISIBLE_DEVICES=1,4 /data/ysli/Johnbo/Focus_High_ABMIL/train_dist.sh 2',
    'output_path': '/data/ysli/Johnbo/Focus_High_ABMIL/results',
    'results_file': 'lr_experiment_results.json',
}

DEFAULT_LR_GRID = ['8e-5', '1e-4', '2e-4', '5e-4', '8e-4', '1e-3', '2e-3', '5e-3']
DEFAULT_WD_GRID = ['0.0', '1e-3', '1e-2']

class ExperimentTracker:
    def __init__(self, results_file):
        self.results_file = results_file
        self.results = self.load_results()

    def load_results(self):
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return []

    def save_results(self):
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def add_result(self, experiment_id, config, metrics):
        result = {
            'experiment_id': experiment_id,
            'config': config,
            'metrics': metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.results.append(result)
        self.save_results()

    def get_best_result(self, metric='test_auc'):
        valid_results = [r for r in self.results if metric in r['metrics']]
        if not valid_results:
            return None
        return max(valid_results, key=lambda x: x['metrics'][metric])

    def get_next_experiment(self, config_grid, continue_from_index=None):
        if continue_from_index is not None:
            if continue_from_index < len(config_grid):
                return continue_from_index, config_grid[continue_from_index]
            return None, None
        run_configs = [str(r['config']) for r in self.results]
        for i, config in enumerate(config_grid):
            if str(config) not in run_configs:
                return i, config
        return None, None

def build_command(base_cmd, config):
    # Parse the base command
    parts = base_cmd.split()

    # Extract environment variable (first part if it contains =)
    env_var = None
    if '=' in parts[0]:
        env_var = parts[0]
        parts = parts[1:]

    # Extract the script path (first remaining part)
    script = parts[0]
    script_args = parts[1:]  # arguments to the script (like '2')

    # Build the command: bash -c "source conda env && bash script <args> --lr=... --weight_decay=..."
    full_cmd = f"source /data/ysli/anaconda3/bin/activate DD && bash {script} {' '.join(script_args)} --lr={config['lr']} --weight_decay={config['wd']}"
    cmd = ['bash', '-c', full_cmd]

    return env_var, cmd

def parse_list(list_str):
    if list_str:
        return [float(x) for x in list_str.split(',')]
    return DEFAULT_LR_GRID

def build_config_grid(lr_list, wd_list):
    configs = []
    for i, lr in enumerate(lr_list):
        for j, wd in enumerate(wd_list):
            config = {
                'lr': lr,
                'wd': wd,
                'experiment_id': f"lr_{i}_wd_{j}",
            }
            configs.append(config)
    return configs

def parse_log_metrics(log_file):
    metrics = {}
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            
            for line in reversed(lines):
                if 'Epoch' in line and ('train loss' in line or 'test loss' in line):
                    parts = line.split(',')
                    for part in parts:
                        if 'accuracy:' in part:
                            metrics['test_acc'] = float(part.split(':')[1].strip())
                        elif 'auc_value:' in part:
                            metrics['test_auc'] = float(part.split(':')[1].strip())
                        elif 'train loss:' in part:
                            loss_str = part.split(':')[1].strip()
                            if 'E' in loss_str:
                                metrics['train_loss'] = float(loss_str.split('E')[0])
                    if 'EarlyStopping' in line:
                        metrics['early_'] = True
                        break
    except:
        pass
    return metrics

def run_single_experiment(config, output_dir):
    print("\n" + "="*80)
    print(f"Start {config['experiment_id']}")
    print("="*80)
    print(f"lr: {config['lr']}")
    print(f"wd: {config['wd']}")
    print("="*80)

    exp_dir = os.path.join(output_dir, config['experiment_id'])
    os.makedirs(exp_dir, exist_ok=True)

    env, cmd = build_command(DEFAULT_CONFIG['base_cmd'], config)

    print(f"Working directory: {exp_dir}")
    if env:
        print(f"Env: {env}")
    print(f"Command: {' '.join(cmd)}")

    start_time = datetime.now()
    return_code = 0
    metrics = {}

    # Set environment if provided
    if env:
        env_parts = env.split()
        for part in env_parts:
            if '=' in part:
                key, value = part.split('=', 1)
                os.environ[key] = value

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd='/data/ysli/Johnbo/Focus_High_ABMIL',
            universal_newlines=True
        )

        log_file = os.path.join(exp_dir, 'training.log')
        with open(log_file, 'w') as f:
            while True:
                line = process.stdout.readline()
                if line == '' and process.poll() is not None:
                    break
                if line:
                    f.write(line)
                    print(line.rstrip())

        process.wait()
        return_code = process.returncode

    except KeyboardInterrupt:
        print("\nThe experiment was interrupted.")
        return_code = -1
    except Exception as e:
        print(f"Error: {e}")
        return_code = -2

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    if return_code == 0:
        metrics.update(parse_log_metrics(log_file))

    # Handle 'N/A' format
    test_auc = metrics.get('test_auc', 'N/A')
    test_acc = metrics.get('test_acc', 'N/A')
    auc_str = f"{test_auc:.4f}" if test_auc != 'N/A' else 'N/A'
    acc_str = f"{test_acc:.4f}" if test_acc != 'N/A' else 'N/A'

    print(f"\n {config['experiment_id']} Completed, time taken: {duration/60:.1f} min.")
    print(f"test AUC: {auc_str}")
    print(f"test ACC: {acc_str}")

    return metrics

def print_summary(tracker):
    print("\n" + "="*80)
    print("Experiment Summary")
    print("="*80)

    if not tracker.results:
        print("No experimental results")
        return

    sorted_results = sorted(
        [r for r in tracker.results if 'test_auc' in r['metrics']],
        key=lambda x: x['metrics']['test_auc'],
        reverse=True
    )

    print(f"\Number of experiments completed: {len(tracker.results)}")
    print(f"\n{'rank':<5} {'ID':<20} {'lr':<12} {'wd':<12} {'test AUC':<10} {'test ACC':<12}")
    print("-" * 80)

    for i, result in enumerate(sorted_results):
        metrics = result['metrics']
        config = result['config']
        print(f"{i+1:<5} {result['experiment_id']:<20} {config['lr']:<12.1e} {config['wd']:<12.1e} "
              f"{metrics.get('test_auc', 0):.4f} {metrics.get('test_acc', 0):.4f} "
              f"{metrics.get('train_loss', 0):.4f} {metrics.get('duration_minutes', 0):.2f}")

    best = tracker.get_best_result('test_auc')
    if best:
        print(f"\nbest experiment: {best['experiment_id']}")
        print(f"config: LR={best['config']['lr']}, WD={best['config']['wd']}")
        print(f"test AUC: {best['metrics']['test_auc']:.4f}")
        print(f"test ACC: {best['metrics']['test_acc']:.4f}")
        print(f"train_loss: {best['metrics']['train_loss']:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_list', type=str, default=None)
    parser.add_argument('--wd_list', type=str, default='1e-3,1e-2')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--continue_from', type=int, default=None)

    args = parser.parse_args()

    results_file = os.path.join(DEFAULT_CONFIG['output_path'], DEFAULT_CONFIG['results_file'])

    tracker = ExperimentTracker(results_file)

    if args.resume:
        tracker.results_file = args.resume
        tracker.results = tracker.load_results()

    lr_list = parse_list(args.lr_list)
    wd_list = parse_list(args.wd_list)
    config_grid = build_config_grid(lr_list, wd_list)

    if args.continue_from is not None:
        idx, config = tracker.get_next_experiment(config_grid, args.continue_from)
        if idx is not None:
            metrics = run_single_experiment(config, DEFAULT_CONFIG['output_path'])
            tracker.add_result(config['experiment_id'], config, metrics)
        else:
            print("All Completed.")
            print_summary(tracker)
            return
    else:
        print(f"\ntotal experiments: {len(config_grid)} ")

        completed = 0
        for i, config in enumerate(config_grid):
            print(f"\nnow: {i+1}/{len(config_grid)}")

            metrics = run_single_experiment(config, DEFAULT_CONFIG['output_path'])
            tracker.add_result(config['experiment_id'], config, metrics)
            completed += 1

            best = tracker.get_best_result('test_auc')
            if best:
                print(f"\nnow best: {best['experiment_id']}, AUC={best['metrics']['test_auc']:.4f}")

    print_summary(tracker)

    print(f"Continue the experiment using the following command:")
    print(f"  python auto_lr_experiment.py --resume {results_file}")

if __name__ == '__main__':
    main()
