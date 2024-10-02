# Large-Language-Models

# **Large Language Models (LLM) Project**

## **What are Large Language Models and Why They Are Important**

Large Language Models (LLMs) are a class of machine learning models designed to understand, generate, and manipulate natural language. These models are trained on vast amounts of text data and use complex architectures like Transformers to perform various tasks such as translation, summarization, and answering questions.

### **Why LLMs are Important:**
- They provide a way to scale language-based solutions across multiple industries.
- LLMs power many of today's intelligent systems such as chatbots, language translation tools, and content generation platforms.
- Fine-tuning LLMs allows for domain-specific applications like legal document analysis, healthcare diagnostics, etc.

---

## **Big Data Problems: Characteristics**

- Large volumes of data (terabytes or petabytes).
- Requires high computational power for processing.
- High variability in the data (text, image, audio).
- Complex relationships within data.
  
## **Small Data Problems: Characteristics**

- Limited data availability (megabytes or gigabytes).
- Difficult to train sophisticated models due to lack of diverse information.
- Requires transfer learning or data augmentation techniques to improve accuracy.

---

## **The Rise of Deep Learning in Natural Language Processing (NLP)**

### **Historical Milestones:**
- **2001:** Learning probabilities over a sequence of words (e.g., n-grams).
- **2013:** Grouping words with similar meanings based on the context (e.g., Word2Vec).
- **2015:** Preserving word context by assigning weights relative to other words (e.g., Attention mechanisms).
- **2018:** Learning language through exposure to large-scale corpora (e.g., the internet) to develop highly accurate LLMs (e.g., BERT, GPT).

The key idea is to create a **baseline model** from where specific use cases can be developed by fine-tuning or modifying existing architectures.

---

## **Recurrent Neural Networks (RNNs)**

RNNs were developed to solve the problem of input memorization in traditional Neural Networks by using loops to remember sequences over time.

### **Types of RNN Architectures:**

- **One-to-One:** Single input, single output (e.g., Image Classification).
- **One-to-Many:** Single input, sequence output (e.g., Image Captioning).
- **Many-to-One:** Sequence input, single output (e.g., Sentiment Analysis).
- **Many-to-Many:** Sequence input, sequence output (e.g., Machine Translation, Summarization).

---

### **Text Generation with RNNs**

- **Many-to-One (One Character at a Time):** Used for Sentiment Analysis, where a sequence of words is reduced to a sentiment label.
- **Many-to-Many (Full Sentences):** Used for tasks like Machine Translation, where full sentences are input and output in different languages.
- **Many-to-Many (Full Sentences):** Used for Text Summarization, where long input text is reduced to a summary.

---

## **Encoder-Decoder Architecture**

This architecture sets the standard for modern LLMs by using two networks:
- **Encoder:** Processes the input sequence and compresses it into a context vector.
- **Decoder:** Generates the output sequence from the context vector.

**Diagram (Textual Representation):**




This architecture is used in various applications like machine translation and summarization.

---

## **What's Under the Hood of RNNs?**

- **Machine Translation with RNNs:** The use of RNN-based architectures in translation tasks helped model long sequences of words in different languages.
  
- **Problems with RNNs:**
  - **Cannot be trained in parallel:** Due to sequential nature, RNNs must be trained one step at a time.
  - **Long Range Dependency:** RNNs struggle with long sequences as they tend to forget information over time.
  - **Sequential Processing:** RNNs process sequences one word at a time, making them slower than modern architectures.

---

## **Challenges of Understanding Language**

- **Language Complexity:** Words can have multiple meanings depending on context.
- **Beyond Word Sequences:** Merely memorizing word sequences isn't enough; a model must understand the broader context in which words are used.

---

## **The Breakthrough: "Attention Is All You Need"**

The **Transformer architecture** revolutionized NLP with its attention mechanism, allowing models to understand the entire context of a sentence by focusing on relevant parts of the input sequence. This led to faster, more accurate LLMs that can be trained in parallel and handle long-range dependencies efficiently.

**Diagram (Textual Representation):**


# **Attention is All You Need**

---

## **Transformers to the Rescue!**

Transformers are a breakthrough architecture in NLP, designed to solve the shortcomings of RNNs and improve scalability, parallelization, and contextual understanding.

### **Key Advantages of Transformers:**

- **Scale Efficiently:** They handle large datasets without the bottlenecks of RNNs or LSTMs.
- **Parallelization:** Unlike RNNs, transformers process words simultaneously, allowing for faster training.
- **Better Contextual Learning:** The self-attention mechanism allows transformers to understand the context of each word relative to the entire sentence.

---

## **Self-Attention Mechanism:**

Self-attention enables the model to focus on different parts of the input sequence when encoding a specific word.

**Example Sentence:**  
*The teacher taught the student with the book.*

In the self-attention mechanism, each word can "attend" to other words in the sentence. Here's how each word might connect in the self-attention process:

- **The** → [teacher, taught, student, with, book]
- **Teacher** → [the, taught, student, with, book]
- **Taught** → [the, teacher, student, with, book]
- **Student** → [the, teacher, taught, with, book]
- **With** → [the, teacher, taught, student, book]
- **Book** → [the, teacher, taught, student, with]

The self-attention mechanism calculates the relationship between each word and all other words in the sentence to understand the full context.

---

## **Transformer Architecture**

**Diagram:**



- **Encoder:** Processes the entire input sequence and generates a contextual representation for each token.
- **Decoder:** Uses the encoder’s output and generates the final sequence step by step, attending to both previous outputs and encoder’s representation.

---

## **Tokenization in Transformers**

Transformers only work with **numerical input**, which means text has to be converted into numbers using a **tokenizer**.

### **Transforming Text Inputs:**

For example, the sentence **“The teacher taught the student”** is tokenized into numerical representations:

| Token ID | Word      |
|----------|-----------|
| 220      | The       |
| 432      | Teacher   |
| 857      | Taught    |
| 220      | the       |

Each word is mapped to a specific token ID.

---

## **Types of Tokenizers**

### 1. **Word-Based Tokenization**
This approach splits the sentence by words and assigns a unique token to each word.

| Token ID | Word      |
|----------|-----------|
| 220      | The       |
| 432      | Teacher   |
| 857      | Taught    |
| 122      | the       |

Example:
- "Let's Create Tokens now" would be tokenized as:
  
  | Token ID | Word     |
  |----------|----------|
  | 120      | Let's    |
  | 302      | Create   |
  | 417      | Tokens   |
  | 189      | now      |

**Limitations:**
- **Vocabulary Size:** Large models require thousands of unique tokens.
- **Unknown Words (OOV):** Any word not in the vocabulary becomes a problem and is treated as "unknown."

### 2. **Subword-Based Tokenization**
Instead of splitting words fully, it breaks words into **subword units**. This addresses the out-of-vocabulary (OOV) issue by breaking down unfamiliar words into known subword pieces.

For example:
- **UncommonWord** might be split as:
  - `Uncommon` → `Un`, `common`, `Word`

### 3. **Character-Based Tokenization**
This approach treats each character as a token. It can handle any input but results in longer sequences.

For example:
- **"Taught"** would be tokenized as:
  - `T`, `a`, `u`, `g`, `h`, `t`

### **Tokenization Example and Comparison:**

#### **Word-Based:**
| Token ID | Word   |
|----------|--------|
| 220      | The    |
| 432      | Teacher|
| 857      | Taught |

#### **Character-Based:**
| Token ID | Character |
|----------|-----------|
| 12       | T         |
| 45       | h         |
| 17       | e         |

#### **Subword-Based:**
| Token ID | Subword   |
|----------|-----------|
| 220      | Teach     |
| 122      | -er       |

---

## **Conclusion**

Transformers revolutionized NLP by introducing attention mechanisms, allowing models to focus on relevant parts of the input. The tokenization techniques play a crucial role in enabling these models to handle diverse language inputs efficiently. By converting words into numerical representations and applying self-attention, Transformers achieve unparalleled performance in tasks like translation, summarization, and more.

