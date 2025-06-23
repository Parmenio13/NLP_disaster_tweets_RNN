## Introduction to Deep Learning - Week 4 Project
## **Natural Language Processing with Disaster Tweets: RNN Approach**

## **0. Project Topic:**
This Kaggle competition lies in the nuanced nature of language on Twitter, where metaphorical expressions and sarcasm can make automated classification difficult.

## Natural Language Processing with Disaster Tweets: RNN Approach

## **1. Problem and Data Description:**

**Challenge Problem**: This is a binary classification task where we need to determine whether a tweet is about a real disaster (1) or not (0). The challenge lies in the nuanced nature of language on Twitter, where metaphorical expressions and sarcasm can make automated classification difficult.

**NLP Context**: Natural Language Processing (NLP) involves teaching machines to understand human language. For this task, we need to process tweet text, extract meaningful features, and classify them accurately despite the informal nature of Twitter language, abbreviations, and noise.

**Dataset Characteristics**:
- Training set: 9,000 tweets (train.csv)
- Test set: 3,700 tweets (test.csv)
- Features:
  - `id`: Unique identifier
  - `text`: Tweet content (string)
  - `location`: Tweet origin location (string, may be empty)
  - `keyword`: Relevant keyword from tweet (string, may be empty)
  - `target`: Binary label (only in train.csv)

### Data Cleaning Plan
1. Handle missing values in location and keyword
2. Clean tweet text:
   - Remove URLs, mentions, and special characters
   - Convert to lowercase
   - Handle contractions
   - Remove stopwords
   - Perform lemmatization
3. Explore using both keyword/location features along with text

## **3. Model Architecture**

### Preprocessing and Word Embeddings
We'll use GloVe (Global Vectors for Word Representation) embeddings because:
- They capture both global statistics and local semantics of words
- Pre-trained on large corpora, so they understand word relationships
- More efficient than training embeddings from scratch on our small dataset

**GloVe Explanation**:
GloVe creates word vectors by analyzing word co-occurrence probabilities across the entire corpus. It combines the benefits of:
- Global matrix factorization (like LSA)
- Local context window methods (like Word2Vec)

The key idea is that the ratio of word co-occurrence probabilities encodes meaning components.

### RNN Architecture
We'll implement a Bidirectional LSTM network because:
- LSTMs handle long-range dependencies better than simple RNNs
- Bidirectional processing captures context from both directions
- Works well with sequential data like text

  Model: "sequential_3"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ embedding_3 (Embedding)         │ ?                      │     1,588,100 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ bidirectional_6 (Bidirectional) │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ bidirectional_7 (Bidirectional) │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_6 (Dense)                 │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_3 (Dropout)             │ ?                      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_7 (Dense)                 │ ?                      │   0 (unbuilt) │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 1,588,100 (6.06 MB)
 Trainable params: 0 (0.00 B)
 Non-trainable params: 1,588,100 (6.06 MB)

## **4. Results and Analysis**

### Training and Evaluation

### Hyperparameter Tuning
We experimented with:
1. Different embedding dimensions (50, 100, 200)
2. LSTM units (32, 64, 128)
3. Number of LSTM layers (1, 2)
4. Dropout rates (0.3, 0.5)
5. Learning rates (0.001, 0.0001)

Best configuration:
- GloVe 100d embeddings
- Two bidirectional LSTM layers (64 and 32 units)
- Dropout rate of 0.5
- Adam optimizer with default learning rate

### Performance Metrics

48/48 ━━━━━━━━━━━━━━━━━━━━ 2s 24ms/step
              precision    recall  f1-score   support

           0       0.79      0.91      0.85       874
           1       0.85      0.67      0.75       649

    accuracy                           0.81      1523
   macro avg       0.82      0.79      0.80      1523
weighted avg       0.82      0.81      0.81      1523

### Key Findings
1. The model achieved ~80% validation accuracy
2. Overfitting was observed after 5-6 epochs
3. Adding a second LSTM layer improved performance slightly
4. Higher dropout rates helped with generalization
5. Pretrained embeddings performed better than training from scratch

## **5. Conclusion and Future Work**

### Results Interpretation
The bidirectional LSTM with GloVe embeddings performed reasonably well on this task, achieving about 80% accuracy. The model was better at identifying non-disaster tweets (higher precision) than detecting actual disasters (higher recall).

### Key Learnings
1. Pretrained word embeddings significantly boost performance on small datasets
2. Bidirectional processing helps capture context in both directions
3. Dropout is crucial for preventing overfitting in text classification
4. Text cleaning and preprocessing have a substantial impact on results

### Challenges Faced
1. Handling sarcasm and metaphorical language
2. Short text length provides limited context
3. Class imbalance (more non-disaster tweets)

### Future Improvements
1. Try transformer-based models (BERT, RoBERTa)
2. Incorporate metadata (location, keyword) more effectively
3. Experiment with attention mechanisms
4. Use more sophisticated text preprocessing
5. Try data augmentation techniques for the minority class
