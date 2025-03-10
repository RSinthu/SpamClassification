# Spam Detection System

## Overview
This project implements a spam detection system using machine learning techniques. It processes SMS messages and classifies them as either "spam" or "ham" (legitimate messages) using Natural Language Processing (NLP) and a Naive Bayes classifier.

## Features
- Text preprocessing and cleaning
- Feature extraction using TF-IDF
- Classification using Multinomial Naive Bayes
- Performance evaluation

## Dataset
The project uses the "spam.csv" dataset with the following structure:
- v1 (renamed to 'class'): Message classification (spam/ham)
- v2 (renamed to 'sms'): The actual SMS message content

## Dependencies
- Python 3.x
- pandas
- nltk
- scikit-learn
- matplotlib (for visualization)

## Installation
```bash
pip install pandas nltk scikit-learn matplotlib
```

## NLTK Resources
The following NLTK resources are required:
```python
nltk.download('stopwords')
nltk.download('punkt')
```

## Usage
1. Ensure the "spam.csv" dataset is in the same directory as the script
2. Run the script to:
   - Load and preprocess the data
   - Train the Naive Bayes model
   - Evaluate the model's performance

## Code Explanation

### Data Preprocessing
- Loads data from "spam.csv" using latin-1 encoding
- Removes unnecessary columns
- Renames columns for clarity
- Removes duplicate entries
- Adds a 'Length' feature that represents the character count of each message

### Text Cleaning
The `clean_text()` function performs several text preprocessing steps:
1. Converts text to lowercase
2. Tokenizes the text
3. Removes non-alphanumeric characters
4. Removes stopwords and punctuation
5. Applies stemming to reduce words to their root form

### Feature Extraction
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features
- Limits to the top 3000 most relevant features

### Model Training and Evaluation
- Splits data into training (80%) and testing (20%) sets
- Trains a Multinomial Naive Bayes classifier
- Evaluates the model using accuracy score

## Results
The model achieves high accuracy in distinguishing between spam and legitimate messages.

## Future Improvements
- Experiment with different classification algorithms
- Implement cross-validation
- Add model persistence to save and load trained models
- Create a simple API or UI for real-time spam detection

## License
This project is licensed under the MIT License.

