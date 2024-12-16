import json
import matplotlib.pyplot as plt

source = "mt_en" 

# Load the JSON data
input_file = f"/netscratch/dgurgurov/projects2024/mt_lrls/src/checkpoints/transformer_{source}/training_data.json"
with open(input_file, "r") as file:
    data = json.load(file)

# Extract data for plotting
training_updates = data["training_updates"]
validation_results = data["validation_results"]
epoch_summaries = data["epoch_summaries"]

# Prepare data for training and validation loss plots
train_epochs = [entry["epoch"] for entry in epoch_summaries]
train_losses = [float(entry["train_loss"]) for entry in epoch_summaries]
valid_epochs = [entry["epoch"] for entry in validation_results]
valid_losses = [float(entry["valid_loss"]) for entry in validation_results]

# Prepare data for BLEU score plot
valid_bleu = [float(entry.get("valid_bleu", 0)) for entry in validation_results]

# Plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(train_epochs, train_losses, label="Training Loss", marker="o")
plt.plot(valid_epochs, valid_losses, label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend()
plt.grid(True)
plt.savefig(f"/netscratch/dgurgurov/projects2024/mt_lrls/src/checkpoints/transformer_{source}/loss_plot.png")
plt.show()

# Plot validation BLEU scores
plt.figure(figsize=(10, 6))
plt.plot(valid_epochs, valid_bleu, label="Validation BLEU", marker="o", color="green")
plt.xlabel("Epoch")
plt.ylabel("BLEU Score")
plt.title("Validation BLEU Scores")
plt.grid(True)
plt.savefig(f"/netscratch/dgurgurov/projects2024/mt_lrls/src/checkpoints/transformer_{source}/bleu_plot.png")
plt.show()

# Additional plot: Training and Validation Perplexities
if "train_ppl" in epoch_summaries[0] and "valid_ppl" in validation_results[0]:
    train_ppl = [float(entry["train_ppl"]) for entry in epoch_summaries]
    valid_ppl = [float(entry["valid_ppl"]) for entry in validation_results]

    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_ppl, label="Training Perplexity", marker="o", color="purple")
    plt.plot(valid_epochs, valid_ppl, label="Validation Perplexity", marker="o", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.yscale("log")
    plt.title("Training and Validation Perplexities")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/netscratch/dgurgurov/projects2024/mt_lrls/src/checkpoints/transformer_{source}/train_val_perplexity_plot.png")
    plt.show()

print("Plots saved as PNG files.")
