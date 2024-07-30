# **Idea:** To use diffusion model for synthetic data refinement // language reconstruction. 

## Why **Diffusion Models:** 
- don't predict words auto-regressively 
	- which generate text one token at a time, diffusion models can generate multiple tokens simultaneously
- generation of more diverse examples:
- better control:
	- can incorporate additional controls, such as sentiment or syntax, during the denoising process, allowing for more fine-grained control over the generated text
- summary:
	- diffusion models for languages work by iteratively refining noisy text embeddings to generate coherent and controlled text

## Literature:
- [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/pdf/2205.14217) Li et al. (2022)
- [DiffuSeq: seq2seq text generation with diffusion models](https://arxiv.org/pdf/2210.08933) Gong et al. (2023)
- [Latent Diffusion for Language Generation](https://proceedings.neurips.cc/paper_files/paper/2023/file/b2a2bd5d5051ff6af52e1ef60aefd255-Paper-Conference.pdf) Lovelace et al. (2023)
- [Likelihood-Based Diffusion Language Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/35b5c175e139bff5f22a5361270fce87-Paper-Conference.pdf) Gulrajani and Hashimoto (2024)

## Code:
- [Minimal Text Diffusion](https://github.com/madaan/minimal-text-diffusion) = _learns a diffusion model of a given text corpus, allowing to generate text samples from the learned model._
- [Latent Diffusion](https://github.com/justinlovelace/latent-diffusion-for-language) 