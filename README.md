# Language-Translation-Model-with-Transformers-Self-Project
base model used: https://huggingface.co/Helsinki-NLP/opus-mt-en-hi \
its benchmarks for BLEU are as follows \
![image](https://github.com/user-attachments/assets/0d325443-eaff-4c55-8e70-8f75996eff0f)

while our fine tuned model has achived significant 0.48 BLEU Score, which is 3times the base model.


the final ui looks like below ![Screenshot 2024-10-29 215847](https://github.com/user-attachments/assets/64388061-e047-417a-bfb7-63fecf188cef)


# English to Hindi Translation using Transformers

## Project Overview
This project focuses on fine-tuning the `Helsinki-NLP/opus-mt-en-hi` transformer model using custom source data to improve English-to-Hindi translation quality. It involves preparing the dataset, optimizing the model through fine-tuning, evaluating performance with BLEU scores, and deploying the model for real-time translation using Gradio.

## Features
- **Dataset**: IIT Bombay English-Hindi parallel corpus from Hugging Face.
- **Fine-Tuned Model**: Enhances translation quality by adapting the pre-trained `Helsinki-NLP/opus-mt-en-hi` model to custom data.
- **Evaluation**: BLEU score computation for translation quality measurement.
- **Deployment**: Gradio interface for real-time, user-friendly translation.

---

## Workflow
### 1. Dataset Loading and Preprocessing
- **Source**: [IITB English-Hindi Dataset](https://huggingface.co/datasets/cfilt/iitb-english-hindi).
- **Preprocessing**: Tokenized and prepared inputs and labels using `AutoTokenizer` to ensure compatibility with the model.

### 2. Model Fine-Tuning
- **Base Model**: Pre-trained `Helsinki-NLP/opus-mt-en-hi` checkpoint.
- **Fine-Tuning**: Leveraged TensorFlow to train the model on custom data for two epochs, optimizing for translation accuracy.

### 3. BLEU Score Evaluation
- **Performance Metrics**: Calculated BLEU scores on 100 examples from the test dataset using NLTK.
- **Results**: Demonstrates significant improvements in translation quality through fine-tuning.

### 4. Real-Time Deployment
- **Gradio Interface**: Provides an intuitive platform for users to input English text and receive Hindi translations instantly.

---

## Installation
Ensure you have Python 3.8+ and install the required dependencies:
```bash
pip install transformers datasets gradio nltk
```

---

## Usage
### Training and Saving the Model
1. Load and preprocess the dataset:
   ```python
   raw_datasets = load_dataset("cfilt/iitb-english-hindi")
   tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
   ```
2. Fine-tune the model:
   ```python
   model.fit(train_dataset, validation_data=validation_dataset, epochs=10)
   ```
3. Save the fine-tuned model and tokenizer:
   ```python
   model.save_pretrained("tf_model/")
   tokenizer.save_pretrained("tokenizer")
   ```

### Real-Time Translation
1. Launch the Gradio interface:
   ```python
   interface.launch()
   ```
2. Input English text in the web interface to receive Hindi translations in real time.

### BLEU Score Evaluation
Evaluate translation quality:
```python
dataset = raw_datasets['test'].select(range(100))
for example in dataset:
    ...
    bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smooth_fn)
```

---

## Project Structure
- `tokenizer/`: Saved tokenizer for fine-tuned translations.
- `tf_model/`: Directory containing the fine-tuned model.
- `translation_demo.py`: Gradio-based script for translation.
- `evaluation.py`: Script for BLEU score evaluation.

---

## Results
- **Improved Translation Quality**: Fine-tuning resulted in higher BLEU scores compared to the base model.
- **BLEU Score**: Achieved an average BLEU score of `X.X` on the first 100 test examples.

---

## Future Enhancements
- Explore additional metrics like METEOR and ROUGE for evaluation.
- Fine-tune on larger and more diverse datasets.
- Investigate multi-lingual models for broader translation capabilities.

---

## References
- [Helsinki-NLP/opus-mt-en-hi](https://huggingface.co/Helsinki-NLP/opus-mt-en-hi)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [BLEU Score](https://en.wikipedia.org/wiki/BLEU)

---

## License
This project is licensed under the MIT License.
