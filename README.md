# DistilBERT Fine-tuning on Emotion Classification Dataset

This repository contains a PyTorch and Transformers implementation for fine-tuning the DistilBERT base uncased model on the `dair-ai/emotion` dataset. The pipeline includes downloading the dataset, loading the pretrained model, training, saving the model and checkpoints, validating, evaluating, resuming training from a checkpoint, and performing inference.

## Features

- Fine-tune DistilBERT on the `dair-ai/emotion` dataset.
- Save and load model checkpoints.
- Resume training from the last saved checkpoint.
- Perform inference on custom text inputs.
- Use PyTorch with CUDA support.
- Includes a `requirements.txt` for dependencies.

---

## Installation

### Prerequisites
Ensure you have Python 3.10+ installed.

### Install PyTorch
To install PyTorch, visit the [official PyTorch website](https://pytorch.org/get-started/locally/) and follow the instructions to select the appropriate version for your system and CUDA setup.

### Install Dependencies
Run the following command to install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Training the Model

### Steps
1. Clone this repository:
    ```bash
    git clone https://github.com/Rahul2991/DistilBert-Based-Emotion-Classification.git
    cd DistilBert-Based-Emotion-Classification
    ```

2. Start training:
    ```bash
    python train.py
    ```

3. Training output includes:
    - Checkpoints: Saved in the `checkpoints/` directory.
    - Final model: Saved in `results/` directory.

4. Resume training from a checkpoint:
    ```bash
    python train.py --resume_from_checkpoint checkpoints/<checkpoint_folder>
    ```

---

## Evaluation and Validation

After training, you can validate the model on the test dataset:

```bash
python evaluate_model.py
```

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

### Results
- Accuracy: 0.9210
- Precision: 0.9218
- Recall: 0.9210
- F1 Score: 0.9212

---

## Inference

To perform inference on custom text:

1. Run the inference script:
    ```bash
    python inference.py
    ```

2. The script will output the predicted emotion label.

---

## Directory Structure

```plaintext
.
├── train.py             # Script for training the model
├── evaluate_model.py    # Script for evaluating the model
├── inference.py         # Script for running inference
├── requirements.txt     # Dependencies
├── results/             # Directory for saving the final model
├── checkpoints/         # Directory for saving training checkpoints
└── README.md            # Project documentation
```

---

## Example Usage

### Training
```bash
python train.py
```

### Resuming Training
```bash
python train.py --resume_training 1 --resume_training_checkpoint checkpoints/checkpoint-10000
```

### Evaluation
```bash
python evaluate_model.py
```

### Inference
```bash
python inference.py
```

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Hugging Face for the `transformers` and `datasets` libraries.
- `dair-ai/emotion` dataset for providing labeled emotion data.
- PyTorch for the deep learning framework.

---

## Citations
```bash
@inproceedings{saravia-etal-2018-carer,
    title = "{CARER}: Contextualized Affect Representations for Emotion Recognition",
    author = "Saravia, Elvis  and
      Liu, Hsien-Chi Toby  and
      Huang, Yen-Hao  and
      Wu, Junlin  and
      Chen, Yi-Shin",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1404",
    doi = "10.18653/v1/D18-1404",
    pages = "3687--3697",
    abstract = "Emotions are expressed in nuanced ways, which varies by collective or individual experiences, knowledge, and beliefs. Therefore, to understand emotion, as conveyed through text, a robust mechanism capable of capturing and modeling different linguistic nuances and phenomena is needed. We propose a semi-supervised, graph-based algorithm to produce rich structural descriptors which serve as the building blocks for constructing contextualized affect representations from text. The pattern-based representations are further enriched with word embeddings and evaluated through several emotion recognition tasks. Our experimental results demonstrate that the proposed method outperforms state-of-the-art techniques on emotion recognition tasks.",
}
```
