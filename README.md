# Low-Resource Machine Translation with Intermediary Language from Language Reconstruction Model

## Project Overview

This project potentially addresses the challenge of translating between low-resource languages (LRLs) using an intermediary language, leveraging existing data and models. The approach involves a multi-stage process where a low-resource machine translation (MT) model is improved by incorporating synthetic data generated through intermediary translations. 

### Approach

1. **Initial Translation**: Use an English-to-LRL MT model to translate a portion of parallel data (PD<sub>firsthalf</sub>) to an intermediary "BadLRL" representation.
2. **Model Training for Refinement**: Train an MT model to translate from "BadLRL" to LRL, aiming to refine the quality of translations from "BadLRL" to "good" LRL.
    - Generate synthetic "BadLRL" data by scrambling LRL sentences or translating them incorrectly from English.
3. **Synthetic Data Generation**: Utilize the two previous systems to create synthetic parallel data (PD<sub>synthetic</sub>).
4. **Final Model Training**: Train the final MT model using both the second half of the parallel data (PD<sub>secondhalf</sub>) and the synthetic data (PD<sub>synthetic</sub>).

### Schematics

![BadLRL Process](assets/badlrl.png)

## Literature Review

### Most Relevant Sources

1. **Improving Neural Machine Translation Models with Monolingual Data**  
   Sennrich et al. (2016)  
   - **Key Contribution**: Introduces the technique of back-translation, where monolingual data is translated into the source language and then used as synthetic parallel data to enhance MT models. This technique is crucial for generating additional training data in scenarios with limited parallel corpora.

2. **Bi-Directional Differentiable Input Reconstruction for Low-Resource Neural Machine Translation**  
   Niu et al. (2019)  
   - **Key Contribution**: Proposes a bi-directional NMT model that learns to reconstruct the original input from the translation. This approach helps better utilize limited parallel data and improve translation quality, which is relevant for enhancing the "BadLRL to LRL" model.

3. **Trivial Transfer Learning for Low-Resource Neural Machine Translation**  
   Kocmi and Bojar (2018)  
   - **Key Contribution**: Describes the transfer learning approach where a well-trained high-resource MT model is adapted to low-resource languages. This method can be applied to initialize and fine-tune MT models in our approach.

4. **Iterative Back-Translation for Neural Machine Translation**  
   Hoang et al. (2018)  
   - **Key Contribution**: Extends basic back-translation by iterating the process to progressively improve synthetic data and translation models. This iterative approach can enhance the quality of synthetic data generated in our multi-stage process.

### Additional Relevant Sources

5. **Understanding Back-Translation at Scale**  
   Edunov et al. (2018)  
   - **Key Contribution**: Analyzes the effects of different back-translation techniques and strategies, including sampling and noise addition, which can inform the synthetic data generation process in this project.

6. **Enhancement of Encoder and Attention Using Target Monolingual Corpora in Neural Machine Translation**  
   Imamura et al. (2018)  
   - **Key Contribution**: Proposes methods to enhance MT models using target language monolingual data. This approach is useful for improving the quality of synthetic data.

7. **Reusing a Pretrained Language Model on Languages with Limited Corpora for Unsupervised NMT**  
   Chronopoulou et al. (2020)  
   - **Key Contribution**: Explores the fine-tuning of pre-trained language models for low-resource languages, which can be relevant for initializing the "BadLRL to LRL" model.

8. **Copied Monolingual Data Improves Low-Resource Neural Machine Translation**  
   Currey et al. (2017)  
   - **Key Contribution**: Discusses copying monolingual data to the source side, which is a technique that can be applied to improve the quality of the generated synthetic data.

9. **Neural Proto-Language Reconstruction**  
   Cui et al. (2024)  
   - **Key Contribution**: Introduces VAE-Transformer for proto-language reconstruction and data augmentation. While focused on proto-languages, the techniques for data augmentation and latent space representation can inspire similar methods for synthetic data generation in our project.

10. **Simple and Effective Noisy Channel Modeling for Neural Machine Translation**  
    Yee et al. (2019)  
    - **Key Contribution**: Explores noisy channel models to utilize unpaired data, which can inform approaches for handling synthetic data and improving translation models.

11. **A Survey on Text Generation Using Generative Adversarial Networks**  
    De Rosa et al. (2019)  
    - **Key Contribution**: Reviews GAN-based methods for text generation, which could provide insights into generating high-quality synthetic data.

## Data Sources

- **Parallel Corpus for English-MT**: [ParaCrawl](https://live.european-language-grid.eu/catalogue/corpus/7072)

