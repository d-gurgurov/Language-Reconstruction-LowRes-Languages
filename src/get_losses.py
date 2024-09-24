from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import argparse

# parsing command line arguments
parser = argparse.ArgumentParser(description='Testing a model on Tatoeba')
parser.add_argument('--language', type=str, required=True, help='Target language code (e.g., "swh" for Swahili)')
parser.add_argument('--model', type=str, required=True, help='Model configuration')
parser.add_argument('--helsinki', type=bool, required=False, default=False, help='Use Helsinki-NLP model?')
args = parser.parse_args()

# setting language code from command line argument
language = args.language
model_type = args.model
lang_map = {"sw": "swahili", "mt": "maltese", "ga": "irish", "is": "icelandic",
            "tl": "tagalog", "hr": "croatian", "nn": "norwegian"}

def load_tensorboard_data(logdir):
    event_acc = event_accumulator.EventAccumulator(logdir)
    event_acc.Reload()
    return event_acc

def plot_metrics(ax, event_acc, metric_names, title, color, styles=None):
    for metric_name in metric_names:
        steps = [event.step for event in event_acc.scalars.Items(metric_name)]
        values = [event.value for event in event_acc.scalars.Items(metric_name)]

        # Define line style and width based on the metric name
        style = styles.get(metric_name, '-') if styles else '-'
        line_width = 2.5 if 'train' in metric_name else 1.5  # Thicker line for 'train'

        ax.plot(steps, values, label=metric_name, color=color, linestyle=style, linewidth=line_width)

    ax.set_title(title)
    ax.set_xlabel('Steps')
    ax.legend()

def visualize_tensorboard_logs(logdir, save=False, save_path=None):
    event_acc = load_tensorboard_data(logdir)

    print("Available scalar metrics (tags):")
    print(event_acc.scalars.Keys())

    # Specify the metrics to plot
    metrics = ['train/loss', 'eval/loss'] 

    # Create subplots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define color for the plot
    color = 'blue'

    # Define different line styles for train and eval losses
    styles = {
        'train/loss': '-',  # Solid line for training loss (bold)
        'eval/loss': ':'    # Dotted line for evaluation loss
    }

    # Plot the metrics
    plot_metrics(ax, event_acc, metrics, 'Training Metrics', color, styles)

    ax.set_ylim(top=0.5)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig(save_path, pad_inches=0.1, bbox_inches='tight', dpi=100)


if __name__ == "__main__":
    logdir = f'/netscratch/dgurgurov/projects2024/mt_lrls/models/{lang_map[language]}/{model_type}/results/runs'

    visualize_tensorboard_logs(logdir, save=True, save_path=f"{lang_map[language]}_{model_type}.png")
