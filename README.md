# LegalDocSummarizer

Here is the README file content for your GitHub repository, formatted in Markdown and including the important images from the document.

# Legal Text Question Answering and Summarization Using Reinforcement Learning

[cite_start]This project, developed by Pranav Moothedath, Abhishek Srinivas, Shreesha M, and Arnav Santosh from the Department of Artificial Intelligence, NITK Surathkal, introduces a system for summarizing Indian legal texts and answering questions related to them. [cite: 1, 2, 3] [cite_start]Given the complexity and length of legal documents in India, this tool aims to make legal information more accessible to the general public. [cite: 5]

## Abstract

[cite_start]The system tackles two primary tasks: summarization and question answering for Indian legal documents. [cite: 7] [cite_start]For summarization, a model is trained using reinforcement learning to generate accurate and meaningful summaries. [cite: 8] [cite_start]This approach rewards the model for producing outputs that are relevant, concise, and similar to summaries written by experts, allowing it to learn and improve over time. [cite: 9] [cite_start]The question-answering functionality uses the Pinecone Database and Groq's LLaMA3 language model. [cite: 11] [cite_start]The system's effectiveness is demonstrated through evaluations using ROUGE and NLI metrics, showing strong performance in creating accurate summaries and clear answers for lengthy legal documents. [cite: 12, 13, 14]

## Introduction

[cite_start]The Indian legal system generates a vast number of complex documents that are often difficult for most people to understand due to their specialized language and convoluted sentence structures. [cite: 18, 19] [cite_start]With the increasing digitization of court cases, there is a growing need for automated systems that can process and simplify these legal texts. [cite: 21] [cite_start]This project addresses these challenges by developing a system to summarize legal documents and answer specific questions about them. [cite: 23, 24] [cite_start]For summarization, we employ abstractive techniques, where the model generates a summary in its own words, a method more effective for capturing the nuances of legal texts than extractive summarization. [cite: 25, 26, 27]

## Methodology

The project is divided into two independent systems: Legal Summarization and Legal Question Answering.

### System I: Legal Summarization

[cite_start]This system fine-tunes a legal domain language model using Reinforcement Learning (RL) with a Parameter-Efficient Fine-Tuning (PEFT) technique called LoRA (Low-Rank Adaptation). [cite: 81, 91]

* [cite_start]**Base Model**: `nsi319/legal-led-base-16384`, a pre-trained legal domain seq-to-seq transformer model capable of handling long documents up to 16,384 input tokens. [cite: 83, 84]
* [cite_start]**Dataset**: The Indian Legal Dataset (ILDC) from Hugging Face, which includes 7,030 training samples and 1,000 testing samples with high-quality reference summaries from legal professionals. [cite: 31, 85, 86]
* [cite_start]**PEFT/LoRA Implementation**: To improve training efficiency, LoRA was used with a rank of 8, alpha of 16, and a dropout of 0.1, targeting the attention and feedforward layers of the model. [cite: 91, 93, 94, 95, 96, 97]
* **Reward Function**: The RL process is guided by a reward function calculated as a weighted average of three scores:
    * [cite_start]**Entailment Score (E)**: Measures how much the generated summary is supported by the original text, calculated using cosine similarity of their embeddings. [cite: 113, 134]
    * [cite_start]**Kullback-Leibler (KL) Divergence (KL)**: Ensures the fine-tuned model does not stray too far from the base model's language patterns. [cite: 137, 138]
    * [cite_start]**Length Penalty (L)**: Discourages the generation of overly short summaries. [cite: 139]
    [cite_start]The net reward is formulated as: `Net Reward = 0.6*E + 0.3*KL + 0.1*L`. [cite: 112]

The overall methodology for the summarization system is illustrated below:

[cite_start]![Methodology for Fine-Tuning Legal-LED with LoRA + RL](https://storage.googleapis.com/gemini-prod/images/051794b6-0e1d-4033-ae8f-4ed7927d61c7.png) [cite: 114]

### System II: Legal Question Answering

[cite_start]A Retrieval-Augmented Generation (RAG) approach is used for the question-answering system. [cite: 182]

* **System Architecture**:
    * [cite_start]**Vector Database**: Pinecone is used for efficient storage and retrieval of document embeddings. [cite: 184, 185]
    * [cite_start]**Embedding Generation**: The `multilingual-e5-large` model generates vector representations for documents and user queries. [cite: 186]
    * [cite_start]**Language Model**: Groq's LLaMA3-70b model is used to generate answers based on the retrieved contexts. [cite: 187]
* [cite_start]**Document Processing**: Legal texts are chunked using a sentence-preserving method, with each chunk being a maximum of 505 tokens. [cite: 188, 191, 225] [cite_start]These chunks are then converted to embeddings and stored in Pinecone. [cite: 192, 193]
* **Query Processing**: The system supports two workflows:
    1.  [cite_start]**Regular Questions**: For general legal questions, the system checks a cache for similar past queries before searching the entire legal knowledge base. [cite: 233, 234]
    2.  [cite_start]**Questions on New Documents**: When a user uploads a new document, it is processed and stored in a dedicated namespace for targeted question answering. [cite: 236, 237]

The question-answering workflows are depicted in the following diagram:

[cite_start]![Methodology of Legal Question Answering](https://storage.googleapis.com/gemini-prod/images/5d326f50-d475-4303-926f-4c5414ab26a0.png) [cite: 224]

### Web-Based Frontend Tool

[cite_start]A user-friendly web interface was developed using FastAPI for the backend and NextJS for the frontend, allowing users to upload documents, request summaries, and ask questions in real-time. [cite: 238, 239, 240, 242]

[cite_start]![Frontend Interface for Legal Summary](https://storage.googleapis.com/gemini-prod/images/e775a610-85f2-45a7-9388-75c1a79f831b.png) [cite: 253]

## Results and Analysis

### System I: Legal Document Summarization

[cite_start]The fine-tuned model showed significant improvements over the base model in summarizing legal texts. [cite: 257]

* **Performance Metrics**: The LoRA-fine-tuned model demonstrated superior performance across various metrics, including entailment, factual accuracy, and completeness. [cite_start]The optimal performance was achieved with a LoRA rank of 8 and a KL divergence weight of 0.3. [cite: 333]

    [cite_start]![Performance Across Multiple Metrics](https://storage.googleapis.com/gemini-prod/images/5089e5f5-a6a9-4504-83e1-73934d402b93.png) [cite: 264]

* [cite_start]**Entailment and Summary Length**: The fine-tuned model consistently achieved higher entailment scores while maintaining a more appropriate summary length compared to the base model. [cite: 308]

    [cite_start]![Steps vs Entailment](https://storage.googleapis.com/gemini-prod/images/7376c039-b9d9-482f-893c-28b9d3b10c9c.png) [cite: 258, 290]
    [cite_start]![Steps vs Summary Length](https://storage.googleapis.com/gemini-prod/images/15049383-c5eb-4a25-a745-090c8a6fcf7c.png) [cite: 303]

* [cite_start]**Efficiency**: Using LoRA reduced the number of trainable parameters to just 0.8% of the original model, decreasing the training memory requirement from 24.8 GB to 9.8 GB and increasing the training speed by nearly four times. [cite: 313, 314]

    [cite_start]![LoRA Memory and Speed Comparison](https://storage.googleapis.com/gemini-prod/images/41d63637-23b0-466d-88f6-df3228d4e9fd.png) [cite: 316]

* [cite_start]**Performance on Different Document Lengths**: The fine-tuned model showed increasingly better performance over the base model as the input document length increased, with a 31.2% improvement for documents over 4,000 tokens. [cite: 386, 388]

    [cite_start]![Performance by Document Length](https://storage.googleapis.com/gemini-prod/images/0c0428be-67d4-42b7-a3ca-79a405a81eb8.png) [cite: 336]

* [cite_start]**Hyperparameter Tuning**: A heatmap analysis of different LoRA ranks and KL divergence weights confirmed that a rank of 8 and a KL weight of 0.3 provided the best performance. [cite: 333]

    [cite_start]![KL divergence weight against LoRA Rank](https://storage.googleapis.com/gemini-prod/images/3a9a5a3a-5cc6-47b8-b80c-e2f7b8f04fd6.png) [cite: 383]

### System II: Legal Question Answering

[cite_start]The RAG system was evaluated on a dataset of 1,897 unique question-answer pairs from approximately 300 Indian legal cases. [cite: 392]

* **Evaluation Metrics**:
    * [cite_start]**ROUGE Scores**: The system achieved average ROUGE-1, ROUGE-2, and ROUGE-L scores of 0.3, 0.18, and 0.27, respectively. [cite: 398] [cite_start]The scores were impacted by the LLM generating longer answers than the ground truth. [cite: 409]
        [cite_start]![Average ROUGE Scores](https://storage.googleapis.com/gemini-prod/images/7747e928-f66f-4bd6-97b7-6e696f5b90f4.png) [cite: 317]
    * [cite_start]**BLEU Scores**: The BLEU scores varied, with some examples showing high scores (above 0.6) while most were moderate to low. [cite: 413]
        [cite_start]![Individual BLEU Scores](https://storage.googleapis.com/gemini-prod/images/a568c0b5-7c15-46c5-84ac-a4e1d93962d2.png) [cite: 401]
    * [cite_start]**Semantic Similarity**: The average cosine similarity score between generated and ground-truth answers was approximately 0.4. [cite: 416, 417]
        [cite_start]![Answer Relevancy Scores](https://storage.googleapis.com/gemini-prod/images/73fdfb0d-d4a6-444f-b645-8126788523c1.png) [cite: 406]
    * [cite_start]**Precision, Recall, and F1 Scores**: The average precision, recall, and F1-score were 0.32, 0.36, and 0.338, respectively. [cite: 422]
        [cite_start]![Precision, Recall, and F1 Scores](https://storage.googleapis.com/gemini-prod/images/2237599c-e35f-4054-bc7f-273b5f92ffb7.png) [cite: 402]
    * [cite_start]**System Efficiency**: The document upload time showed a linear relationship with the number of chunks, while the RAG processing time remained relatively constant at around 1.15 seconds, indicating excellent query efficiency. [cite: 424, 426]
        [cite_start]![System Efficiency Evaluation](https://storage.googleapis.com/gemini-prod/images/6df2881a-6bb8-45e0-9426-ed87661cf543.png) [cite: 421]

## Conclusion and Future Scope

[cite_start]This project successfully implemented a robust application for legal text summarization and question answering. [cite: 430] [cite_start]The RL-based fine-tuning significantly improved the summarization model's factual coherence and reduced hallucinations. [cite: 431, 434] [cite_start]The RAG system for question answering demonstrated efficient retrieval and generation, although the verbosity of the generated answers affected some metrics. [cite: 436, 437]

Future work will focus on:
* [cite_start]Fine-tuning the summarization transformer for a larger number of epochs, provided more compute power is available. [cite: 435]
* [cite_start]Improving the question-answering module by using user-specific database indices to enhance retrieval. [cite: 439, 440]
* [cite_start]Fine-tuning the LLaMA 3 model specifically for legal question answering to improve accuracy and conciseness. [cite: 441, 443]
