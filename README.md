# Low-Resource Machine Translation with Intermediary Language from Language Reconstruction Model

## Project Overview

This project addresses the challenge of translating between low-resource languages (LRLs) using an intermediary language, leveraging existing data and models. The approach involves a multi-stage process where a low-resource machine translation (MT) model is improved by incorporating synthetic data generated through intermediary translations. 

### Approach

1. **Initial Translation**: Use an English-to-LRL MT model to translate a portion of parallel data (PD<sub>firsthalf</sub>) to an intermediary "BadLRL" representation.
2. **Model Training for Refinement (Reconstruction Model)**: Train an MT model to translate from "BadLRL" to LRL, aiming to refine the quality of translations from "BadLRL" to "good" LRL.
    - Generate synthetic "BadLRL" data by scrambling LRL sentences or translating them incorrectly from English.
3. **Synthetic Data Generation**: Utilize the two previous systems to create re-fined synthetic parallel data (PD<sub>synthetic</sub>).
4. **Final Model Training**: Train the final MT model using both the second half of the parallel data (PD<sub>secondhalf</sub>) and the re-fined synthetic data (PD<sub>synthetic</sub>).

### Prelimenary experiments

- (1) Find parallel corpus for English-Maltese & MT system + 
	- [en-mt](https://live.european-language-grid.eu/catalogue/corpus/7072) from ParaCrawl
	- Helsinki-NLP
- (2) Train a baseline model on this parallel corpus + 
- (3) Translate the parallel corpus to get BadLRL + 
- (4) Train a reconstruction model for BadLRL to GoodLRL +
- (5) Find a monolingual corpus & translate it with Helsinki-NLP to get BadLRL -
	- [tinystories](https://huggingface.co/datasets/roneneldan/TinyStories?row=10)
- (6) Reconstruct the BadLRL from tinystories with the reconstruction model - 
- (7) Train a model with 50% parallel corpus and 50% BadLRL from step 5 -
	- safe-check
- (8) Train a model with 50% parallel corpus and 50% reconstructed BadLRL from step 6 - 
- (9) Compare models from steps 7 and 8 -

### Schematics
<div align="center">
    <img src="assets/badlrl.png" alt="BadLRL Process" width="500" style="display: block; margin: 0 auto;"/>
<div align="center">

## Literature Review

### Most Relevant Sources

1. **[Improving Neural Machine Translation Models with Monolingual Data](https://aclanthology.org/P16-1009.pdf)**  
   Sennrich et al. (2016)  
   - **Key Contribution**: Back-translation original paper.
    1. Pairing monolingual training data with an automatic backtranslation which is treated as additional parallel training data.
    2. Use the trained model to translate monolingual sentences from the target language back into the source language. This creates synthetic parallel data, which can be used to increase the amount of training data.

2. **[Bi-Directional Differentiable Input Reconstruction for Low-Resource Neural Machine Translation](https://aclanthology.org/N19-1043.pdf)**  
   Niu et al. (2019)  
   - **Key Contribution**: Proposes a bi-directional NMT model that learns to reconstruct the original input from the translation.
   	1. The authors propose a bi-directional NMT model that incorporates a reconstruction task to better utilize the limited parallel data.
	2. Input reconstruction. The model learns to translate in both directions (source to target and target to source) and also tries to reconstruct the original input from the translation.
		- The model first translates the input sentence from the source language to the target language.
		- Then, it attempts to reconstruct the original input by translating the target language output back to the source language
	3. This method helps to extract more information from limited parallel data.

3. **[Trivial Transfer Learning for Low-Resource Neural Machine Translation](https://aclanthology.org/W18-6325.pdf)**  
   Kocmi and Bojar (2018)  
   - **Key Contribution**: Describes the transfer learning approach where a well-trained high-resource MT model is adapted to low-resource languages. 
   	1. NMT model for the high-resource language pair is trained until convergence. This model is called “parent”. 
	2. After that, the child model is trained without any restart, i.e. only by changing the training corpora to the low-resource language pair.
	3. The same hyperparameters are used.

4. **[Iterative Back-Translation for Neural Machine Translation](https://aclanthology.org/W18-2703.pdf)**  
   Hoang et al. (2018)  
   - **Key Contribution**: Extends basic back-translation by iterating the process to progressively improve synthetic data and translation models.
    1. Iterative back-translation extends the basic back-translation method by repeating the process multiple times to progressively improve the quality of the synthetic data and the translation models.
    2. Back-translated data is used to build better translation systems in forward and backward directions, which in turn is used to reback-translate monolingual data.

### Additional Relevant Sources

5. **[Understanding Back-Translation at Scale](https://aclanthology.org/D18-1045.pdf)**  
   Edunov et al. (2018)  
   - **Key Contribution**: Analyzes the effects of different back-translation techniques and strategies, including sampling and noise addition, which can inform the synthetic data generation process in this project.
   	1. Back-translation usually uses beam search or greedy search.
	2. Both lead to less-rich translations.
	3. Sampling from the model distribution as well as adding noise to beam search outputs.

6. **[Enhancement of Encoder and Attention Using Target Monolingual Corpora in Neural Machine Translation](https://aclanthology.org/W18-2707.pdf)**  
   Imamura et al. (2018)  
   - **Key Contribution**: Proposes methods to enhance MT models using target language monolingual data. 
    1. Extends the method proposed by Sennrich et al. (2016) to enhance the encoder and attention using target monolingual corpora.
	2. Proposed method generates multiple source sentences by sampling when each target sentence is translated back.
    

7. **[Reusing a Pretrained Language Model on Languages with Limited Corpora for Unsupervised NMT](https://aclanthology.org/2020.emnlp-main.214.pdf)**  
   Chronopoulou et al. (2020)  
   - **Key Contribution**: Explores the fine-tuning of pre-trained language models for low-resource language translation.
   	1. The monolingual LM is fine-tuned on both languages and is then used to initialize a Unsupervised NMT model. 
	2. To reuse the pretrained LM, they have to modify its predefined vocabulary, to account for the new language. 
	3. A novel vocabulary extension method is proposed.

8. **[Copied Monolingual Data Improves Low-Resource Neural Machine Translation](https://aclanthology.org/W17-4715.pdf)**  
   Currey et al. (2017)  
   - **Key Contribution**: Discusses copying monolingual data to the source side.
    1. Proposed copying monolingual data to the source side for low-resource NMT.
	2. Monolingual data on both sides. Target and source are identical.

9. **[Neural Proto-Language Reconstruction](https://arxiv.org/pdf/2404.15690)**  
   Cui et al. (2024)  
   - **Key Contribution**: Introduces VAE-Transformer for proto-language reconstruction and data augmentation. 
    1. Data Augmentation: The authors develop methods to predict missing data in the dataset, which helps improve the model's performance and stability.
    2. VAE-Transformer: They enhance the Transformer model by adding a Variational Autoencoder (VAE) structure. This addition helps the model better capture the relationships between daughter languages and their proto-forms.
		- VAEs create a more regularized latent space compared to standard autoencoders.
	3. Neural Machine Translation: They adapt techniques from neural machine translation to the task of proto-language reconstruction

10. **[Simple and Effective Noisy Channel Modeling for Neural Machine Translation](https://aclanthology.org/D19-1571.pdf)**  
    Yee et al. (2019)  
    - **Key Contribution**: Explores noisy channel models to utilize unpaired data.
        1. Direct models cannot naturally take advantage of unpaired data. 
	    2. Channel model and language model utilized instead.
	    3. Standard sequence to sequence models are a simple parameterization for the channel probability that naturally exploits the entire source.


11. **[A Survey on Text Generation Using Generative Adversarial Networks](https://pdf.sciencedirectassets.com/272206/1-s2.0-S0031320321X00088/1-s2.0-S0031320321002855/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjELL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIGlaW%2BPbggIaiV4tM8MXcRO7ItJl27SLSuhgSUmYR6ukAiEAxdj6GBtxbLmCib0hmTTeA85kZDalYXN5oKJdDlj%2BtoIqvAUIi%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDAM%2FN22EXi0vcBosIyqQBTD2g%2B96KZMotLRnAvsqJcRflPXFpMNRygRGkyF0vEstUd%2BfmBA%2FDb2bZj1TbVq0ejs%2F9CXzFpayC%2B8yGA1WV5zbrZB8HPQbRrAh9Crjrou24N7y%2F7pjsGJBssPAXzPnpKgYMpdTiyxqQOaTQIkpAzUm2Uf2UiSgEUP4BnRloDysRIO71TZet1yWhvse1f%2Fwn5N7oqQab9E2SxLZ6SWmfWP6iKjHlqlWfxyYO%2FkUuBF8QxaPpzQPI%2FezTu%2BVXejkgqWwtsU7gF3ks2%2Fym0E4jfgUJPXPIt1DuSxv48Wx53IAIcmNkzis0T7bhUyrxqsFfX%2FKJrqc%2BlAyCJDobzteSYcmtdibWa4QmIpssnUSNMivehuSf265259izrZAt%2BqlLQ4%2BprY9tWTdTZv4Hk4nLK95%2BTXnAJ01Y%2BiFRAmfhyItIqh978RNuNuWlqNXd0Zx0FnvVH%2Fv7QAxEjZPK2RbX6xf3AMSfGnDSDVPAotnubqhv969Vhz3mTxrI15P%2Fd%2FEht%2BNgG6q4Ny6DgBji0U%2Bwxk%2FD3AuEZiVGblat%2Fz9LgThsTf1w7Bb2q4tX7ZEFsJbEoeA%2F8vVvVImL9zxhj6tY%2FkwTIW%2FMuQQNA%2BdlsdOq3sCF9VWg9%2FRtD3gIpfQZ8qOt8FjAOhTCQmD0hV7nfpw8XcMMB9X8WdCPR9%2Fejcfi8eeZu%2FCfBhqvTrYaF30%2Fd9vx%2FDRhHjrEOdHyEwUZmH53InJIjWJx1q0eL3ym3DdOXmXiTmqMpf%2BAdwobAxm7wtu%2BbMXrhLIu8N9H%2BlKTUd2xh2Jzp3Z3ZrUCmQWi3Xqthn6tljZr2tNR6TybBWuJcqSh9LvhTEG%2BdeYFei%2Bg5b%2BdxC3MNs5f%2Fb8StN51r%2Fs5U%2F6B7rXAyXfAj5hvdiZ7nlHtVXbE4T49fsFLpBGG3LrLtf88gnbZ%2FOwLRaaRZn3ZB32bsZeyDQafggHBh1GvSt4KcJYhZw7DEG5EjfG41Wz%2Fqvv3tK8zCMARsptNCsC4X5V8q7o7Rh78mOl7YlfTZDvlO8t4PYTh7De4SV53YUN3hAnKwcLFEAhciw%2BBXTg6ZTzochONjKrXLJCUgckloAnSkCUZa0%2BRo6eyJ7swxlFtnNw0%2FdneEjDa9cO%2Fqz1KMu6%2BZblAgFoUK7Zym8hDXAXsVbUE5CBu5zm9%2Fg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20210711T224417Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY7DDGYSKY%2F20210711%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=fc805ed6c94d4e3b500fc32ad8b8efb11a4b382cf6cc2b2c95edb5412f7fc417)**  
    De Rosa et al. (2019)  
    - **Key Contribution**: Reviews GAN-based methods for text generation, might give us some insights into generating high-quality synthetic data.

12. **[Neural Machine Translation for Low-resource Languages: A Survey](https://dl.acm.org/doi/pdf/10.1145/3567592)** Panathunga et al. (2023)
    - **Key Contribution**: Reviews most synthetic generation methods for high- and low-resource languages.

## Side questions:
1. (?) Could the "BadLRL to LRL" model  be trained jointly with the main translation model
2. (?) Conditional Variational Auto-Encoders for creating good synthetic corpus 
	- **Encoder**:
		    - Compresses the input data into a latent space
		    - This latent space is represented as a probability distribution (usually Gaussian)
	- **Latent Space**:
		    - A compact representation of the input
		    - Captures important features of the data
	- **Decoder**:
		    - Takes a sample from the latent space and the condition
		    - Generates output data based on these inputs
3. (?) Diffusion-like models for language reconstruction (Refer to _diffusion_lm_)


## Data Sources

- **Parallel Corpus for English-MT**: [ParaCrawl](https://live.european-language-grid.eu/catalogue/corpus/7072)