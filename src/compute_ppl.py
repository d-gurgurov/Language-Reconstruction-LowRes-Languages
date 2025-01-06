import pandas as pd
import random
import torch
from transformers import AutoModelForMaskedLM
from datasets import Dataset
from tqdm import tqdm

# Import necessary classes and functions from the provided code
from abc import ABC
from math import log, exp
from transformers import AutoTokenizer

class Metric(ABC):
    def __init__(self, model, device):
        self.device = device

    def __call__(self, sentences):
        raise NotImplementedError
    
def find_latest_checkpoint(base_path: str, language_code: str) -> str:
    """
    Finds the latest checkpoint directory for the given language.
    """
    lang_path = os.path.join(base_path, language_code)
    checkpoints = [d for d in os.listdir(lang_path) if d.startswith("checkpoint")]
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))  # Sort by checkpoint number
    latest_checkpoint = checkpoints[-1]  # Get the latest checkpoint
    return os.path.join(lang_path, latest_checkpoint)


class PseudoPerplexity(Metric):
    def __init__(self, model, device):
        super().__init__(model, device)
        self.model = model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)

    def __call__(self, sentences):
        assert len(sentences) > 0

        pseudo_perplexities = []
        for sentence in tqdm(sentences, desc="Computing pseudo-perplexity"):
            tokenized_sentence = self.tokenizer.encode(
                sentence, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            num_tokens = tokenized_sentence.shape[-1]

            pseudo_log_likelihood = self.pseudo_log_likelihood(tokenized_sentence)
            pseudo_perplexity = exp(-1 / num_tokens * pseudo_log_likelihood)
            pseudo_perplexities.append(pseudo_perplexity)

        average_pseudo_perplexity = sum(pseudo_perplexities) / len(pseudo_perplexities)
        return {"values": pseudo_perplexities, "average": average_pseudo_perplexity}

    def pseudo_log_likelihood(self, tokenized_sentence):
        pseudo_log_likelihood = 0
        for token_position, original_token_id in enumerate(tokenized_sentence.squeeze()):
            masked_sentence = tokenized_sentence.clone().to(self.device)
            masked_sentence[:, token_position] = self.tokenizer.mask_token_id
            with torch.no_grad():
                output = self.model(input_ids=masked_sentence)
                logits = output.logits.squeeze()
            probabilities = logits[token_position].softmax(dim=0)
            probability = probabilities[original_token_id]
            pseudo_log_likelihood += log(probability)

        return pseudo_log_likelihood


def compute_pseudo_perplexity_parallel(
    model_name, language_code, source_file, target_file, sample_sizes, output_dir, seed=42
):
    # Set random seed
    random.seed(seed)
    torch.manual_seed(seed)

    # Read source and target files
    with open(source_file, "r", encoding="utf-8") as src, open(target_file, "r", encoding="utf-8") as tgt:
        source_sentences = src.readlines()
        target_sentences = tgt.readlines()

    # Check length consistency
    assert len(source_sentences) == len(target_sentences), "Source and target files must have the same number of lines."

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_name = find_latest_checkpoint(model_name, language_code.replace('_', '-'))
    print(model_name, " being used")
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    pseudo_perplexity_metric = PseudoPerplexity(model, device)

    results = {}
    for sample_size in sample_sizes:
        print(f"Processing sample size: {sample_size}...")
        # Take a random sample
        sample_indices = random.sample(range(len(source_sentences)), sample_size)
        source_sample = [source_sentences[i].strip() for i in sample_indices]
        target_sample = [target_sentences[i].strip() for i in sample_indices]

        # Compute pseudo-perplexity for source and target
        source_ppx = pseudo_perplexity_metric(source_sample)
        target_ppx = pseudo_perplexity_metric(target_sample)

        # Print the average pseudo-perplexity for the current sample size
        print(f"Sample size {sample_size}:")
        print(f"  - Syn Avg Pseudo-Perplexity: {source_ppx['average']:.4f}")
        print(f"  - Boosted Avg Pseudo-Perplexity: {target_ppx['average']:.4f}")


        # Save results to a DataFrame
        df = pd.DataFrame(
            {
                "syn": source_sample,
                "boosted": target_sample,
                "syn_pseudo_perplexity": source_ppx["values"],
                "boosted_pseudo_perplexity": target_ppx["values"],
            }
        )
        output_file = f"{output_dir}/pseudo_perplexity_{sample_size}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved results for sample size {sample_size} to {output_file}.")

        # Store average perplexities for summary
        results[sample_size] = {
            "syn_avg_ppx": source_ppx["average"],
            "boosted_avg_ppx": target_ppx["average"],
        }

    return results


if __name__ == "__main__":
    # Parameters
    model_name = "/netscratch/dgurgurov/thesis/src/mlm/model_fine-tune/glot/xlm-r/"
    language_code = "mlt_Latn"
    source_file = "/netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/newscrawl/newscrawl.syn.1m.mt"
    target_file = "/netscratch/dgurgurov/projects2024/mt_lrls/src/data-bin/reconstruct/reordered.newscrawl.boost.1m.mt"
    sample_sizes = [100, 500, 1000, 2500, 5000, 10000]
    output_dir = "/netscratch/dgurgurov/projects2024/mt_lrls/src/results"

    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Run the script
    results = compute_pseudo_perplexity_parallel(
        model_name, language_code, source_file, target_file, sample_sizes, output_dir
    )

    # Print summary of results
    print("\nSummary of Average Pseudo-Perplexities:")
    for size, res in results.items():
        print(f"Sample size {size}: Source Avg PPX = {res['source_avg_ppx']:.4f}, Target Avg PPX = {res['target_avg_ppx']:.4f}")
