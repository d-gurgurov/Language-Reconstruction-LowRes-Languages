# Low-Resource Machine Translation with Intermediary Language from Language Reconstruction Model

## Project Overview

This project addresses the challenge of translating between low-resource languages (LRLs) using an intermediary language, leveraging existing data and models. The approach involves a multi-stage process where a low-resource machine translation (MT) model is improved by incorporating synthetic data generated through intermediary translations. 

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

1. **[Improving Neural Machine Translation Models with Monolingual Data](https://aclanthology.org/P16-1009.pdf)**  
   Sennrich et al. (2016)  
   - **Key Contribution**: Introduces the technique of back-translation, where monolingual data is translated into the source language and then used as synthetic parallel data to enhance MT models. This technique is crucial for generating additional training data in scenarios with limited parallel corpora.

2. **[Bi-Directional Differentiable Input Reconstruction for Low-Resource Neural Machine Translation](https://aclanthology.org/N19-1043.pdf)**  
   Niu et al. (2019)  
   - **Key Contribution**: Proposes a bi-directional NMT model that learns to reconstruct the original input from the translation. This approach helps better utilize limited parallel data and improve translation quality, which is relevant for enhancing the "BadLRL to LRL" model.

3. **[Trivial Transfer Learning for Low-Resource Neural Machine Translation](https://aclanthology.org/W18-6325.pdf)**  
   Kocmi and Bojar (2018)  
   - **Key Contribution**: Describes the transfer learning approach where a well-trained high-resource MT model is adapted to low-resource languages. This method can be applied to initialize and fine-tune MT models in our approach.

4. **[Iterative Back-Translation for Neural Machine Translation](https://aclanthology.org/W18-2703.pdf)**  
   Hoang et al. (2018)  
   - **Key Contribution**: Extends basic back-translation by iterating the process to progressively improve synthetic data and translation models. This iterative approach can enhance the quality of synthetic data generated in our multi-stage process.

### Additional Relevant Sources

5. **[Understanding Back-Translation at Scale](https://aclanthology.org/D18-1045.pdf)**  
   Edunov et al. (2018)  
   - **Key Contribution**: Analyzes the effects of different back-translation techniques and strategies, including sampling and noise addition, which can inform the synthetic data generation process in this project.

6. **[Enhancement of Encoder and Attention Using Target Monolingual Corpora in Neural Machine Translation](https://aclanthology.org/W18-2707.pdf)**  
   Imamura et al. (2018)  
   - **Key Contribution**: Proposes methods to enhance MT models using target language monolingual data. This approach is useful for improving the quality of synthetic data.

7. **[Reusing a Pretrained Language Model on Languages with Limited Corpora for Unsupervised NMT](https://aclanthology.org/2020.emnlp-main.214.pdf)**  
   Chronopoulou et al. (2020)  
   - **Key Contribution**: Explores the fine-tuning of pre-trained language models for low-resource languages, which can be relevant for initializing the "BadLRL to LRL" model.

8. **[Copied Monolingual Data Improves Low-Resource Neural Machine Translation](https://aclanthology.org/W17-4715.pdf)**  
   Currey et al. (2017)  
   - **Key Contribution**: Discusses copying monolingual data to the source side, which is a technique that can be applied to improve the quality of the generated synthetic data.

9. **[Neural Proto-Language Reconstruction](https://arxiv.org/pdf/2404.15690)**  
   Cui et al. (2024)  
   - **Key Contribution**: Introduces VAE-Transformer for proto-language reconstruction and data augmentation. While focused on proto-languages, the techniques for data augmentation and latent space representation can inspire similar methods for synthetic data generation in our project.

10. **[Simple and Effective Noisy Channel Modeling for Neural Machine Translation](https://aclanthology.org/D19-1571.pdf)**  
    Yee et al. (2019)  
    - **Key Contribution**: Explores noisy channel models to utilize unpaired data, which can inform approaches for handling synthetic data and improving translation models.

11. **[A Survey on Text Generation Using Generative Adversarial Networks](https://pdf.sciencedirectassets.com/272206/1-s2.0-S0031320321X00088/1-s2.0-S0031320321002855/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjELL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIGlaW%2BPbggIaiV4tM8MXcRO7ItJl27SLSuhgSUmYR6ukAiEAxdj6GBtxbLmCib0hmTTeA85kZDalYXN5oKJdDlj%2BtoIqvAUIi%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDAM%2FN22EXi0vcBosIyqQBTD2g%2B96KZMotLRnAvsqJcRflPXFpMNRygRGkyF0vEstUd%2BfmBA%2FDb2bZj1TbVq0ejs%2F9CXzFpayC%2B8yGA1WV5zbrZB8HPQbRrAh9Crjrou24N7y%2F7pjsGJBssPAXzPnpKgYMpdTiyxqQOaTQIkpAzUm2Uf2UiSgEUP4BnRloDysRIO71TZet1yWhvse1f%2Fwn5N7oqQab9E2SxLZ6SWmfWP6iKjHlqlWfxyYO%2FkUuBF8QxaPpzQPI%2FezTu%2BVXejkgqWwtsU7gF3ks2%2Fym0E4jfgUJPXPIt1DuSxv48Wx53IAIcmNkzis0T7bhUyrxqsFfX%2FKJrqc%2BlAyCJDobzteSYcmtdibWa4QmIpssnUSNMivehuSf265259izrZAt%2BqlLQ4%2BprY9tWTdTZv4Hk4nLK95%2BTXnAJ01Y%2BiFRAmfhyItIqh978RNuNuWlqNXd0Zx0FnvVH%2Fv7QAxEjZPK2RbX6xf3AMSfGnDSDVPAotnubqhv969Vhz3mTxrI15P%2Fd%2FEht%2BNgG6q4Ny6DgBji0U%2Bwxk%2FD3AuEZiVGblat%2Fz9LgThsTf1w7Bb2q4tX7ZEFsJbEoeA%2F8vVvVImL9zxhj6tY%2FkwTIW%2FMuQQNA%2BdlsdOq3sCF9VWg9%2FRtD3gIpfQZ8qOt8FjAOhTCQmD0hV7nfpw8XcMMB9X8WdCPR9%2Fejcfi8eeZu%2FCfBhqvTrYaF30%2Fd9vx%2FDRhHjrEOdHyEwUZmH53InJIjWJx1q0eL3ym3DdOXmXiTmqMpf%2BAdwobAxm7wtu%2BbMXrhLIu8N9H%2BlKTUd2xh2Jzp3Z3ZrUCmQWi3Xqthn6tljZr2tNR6TybBWuJcqSh9LvhTEG%2BdeYFei%2Bg5b%2BdxC3MNs5f%2Fb8StN51r%2Fs5U%2F6B7rXAyXfAj5hvdiZ7nlHtVXbE4T49fsFLpBGG3LrLtf88gnbZ%2FOwLRaaRZn3ZB32bsZeyDQafggHBh1GvSt4KcJYhZw7DEG5EjfG41Wz%2Fqvv3tK8zCMARsptNCsC4X5V8q7o7Rh78mOl7YlfTZDvlO8t4PYTh7De4SV53YUN3hAnKwcLFEAhciw%2BBXTg6ZTzochONjKrXLJCUgckloAnSkCUZa0%2BRo6eyJ7swxlFtnNw0%2FdneEjDa9cO%2Fqz1KMu6%2BZblAgFoUK7Zym8hDXAXsVbUE5CBu5zm9%2Fg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20210711T224417Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY7DDGYSKY%2F20210711%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=fc805ed6c94d4e3b500fc32ad8b8efb11a4b382cf6cc2b2c95edb5412f7fc417)**  
    De Rosa et al. (2019)  
    - **Key Contribution**: Reviews GAN-based methods for text generation, which could provide insights into generating high-quality synthetic data.

## Data Sources

- **Parallel Corpus for English-MT**: [ParaCrawl](https://live.european-language-grid.eu/catalogue/corpus/7072)