# Machine Learning Approach for Image-based Entity Extraction Using OCR and Transformer Models

## 1. Introduction

This project addresses the challenge of extracting specific entity values from image-based data. The core idea is to extract meaningful textual information from images and use it as context to predict relevant entity values. The project utilizes **Optical Character Recognition (OCR)** to extract text from images and employs a transformer-based model, **FLAN-T5**, for a sequence-to-sequence task aimed at predicting entity values from the extracted text.

## 2. Approach

The solution involves two key components: 
1. Text extraction from images.
2. Fine-tuning a transformer model to generate correct entity values based on the extracted text.

### 2.1 Data Preparation and Text Extraction

The dataset consists of image URLs, entity names (which represent the questions), and the corresponding target entity values. The process involves:

1. **Downloading images**: Images are fetched from the provided URLs.
2. **Applying OCR**: The **EasyOCR** library is used to extract text from each image.
3. **Generating input sentences**: The extracted text is combined with a corresponding question in the form of "What is the [Entity Name]?" This forms the input sentence. The input sentences, paired with target entity values, create the dataset used for training the model.

**Example**:
- If the entity is "Product Price" and the image contains a receipt, the extracted text from the receipt is paired with the question "What is the Product Price?" The answer (the price value) is used in the model training process.

### 2.2 Model Training and Fine-tuning

The **FLAN-T5** pre-trained model is fine-tuned on this dataset. The model takes the OCR-extracted text (context) and the corresponding question (e.g., "What is the Product Price?") as input to predict the correct entity value. The **T5 model**'s sequence-to-sequence architecture is well-suited for this task since it handles both input and output as text.

Key steps include:

1. **Tokenization**: The input sentences (context + question) and target values are tokenized using the T5 tokenizer. Padding tokens are masked to prevent them from affecting the loss function.
   
2. **Data Handling**: The dataset is split into **training (90%)** and **validation (10%)** sets to monitor performance.

3. **Gradient Management**: Techniques like **gradient checkpointing** and **gradient clipping** are applied to optimize memory usage and prevent issues such as gradient explosion.

4. **Fine-tuning**: The model is fine-tuned using the **AdamW optimizer** over several epochs. The weights are adjusted to minimize the loss function.

### 2.3 Model Evaluation and Validation

The model is evaluated after each epoch using the validation set. The steps include:

- The model generates predictions based on the extracted text and corresponding questions.
- These predictions are compared with ground-truth entity values to compute validation loss and accuracy.

By iterating through this process, the model learns to predict entity values from the image-extracted text more accurately. Performance metrics such as validation loss and accuracy are tracked after each epoch.

## 3. Conclusion

This project demonstrates the successful combination of OCR and a transformer-based model to extract and predict entity values from images. The integration of OCR-extracted text with the **T5 model** provides a flexible and accurate approach to entity extraction, even when dealing with complex images.

### Future Improvements:
- Leveraging **advanced OCR techniques** for better text extraction.
- Applying **data augmentation** to handle challenging cases, such as low-quality images.
- **Expanding** the dataset to enhance the model's robustness in real-world applications.

By implementing these improvements, the model can become more effective in handling diverse and challenging scenarios.
