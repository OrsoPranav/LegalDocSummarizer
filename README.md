# Legal Text Question Answering and Summarization Using Reinforcement Learning

Here we introduces a system for summarizing Indian legal texts and answering questions related to them. Given the complexity and length of legal documents in India, this tool aims to make legal information more accessible to the general public.

## Abstract

The system tackles two primary tasks: summarization and question answering for Indian legal documents. For summarization, a model is trained using reinforcement learning to generate accurate and meaningful summaries. This approach rewards the model for producing outputs that are relevant, concise, and similar to summaries written by experts, allowing it to learn and improve over time. The question-answering functionality uses the Pinecone Database and Groq's LLaMA3 language model. The system's effectiveness is demonstrated through evaluations using ROUGE and NLI metrics, showing strong performance in creating accurate summaries and clear answers for lengthy legal documents.

## Introduction

The Indian legal system generates a vast number of complex documents that are often difficult for most people to understand due to their specialized language and convoluted sentence structures. With the increasing digitization of court cases, there is a growing need for automated systems that can process and simplify these legal texts. This project addresses these challenges by developing a system to summarize legal documents and answer specific questions about them. For summarization, we employ abstractive techniques, where the model generates a summary in its own words, a method more effective for capturing the nuances of legal texts than extractive summarization.

## Methodology

The project is divided into two independent systems: Legal Summarization and Legal Question Answering.

### System I: Legal Summarization

This system fine-tunes a legal domain language model using Reinforcement Learning (RL) with a Parameter-Efficient Fine-Tuning (PEFT) technique called LoRA (Low-Rank Adaptation).

* **Base Model**: `nsi319/legal-led-base-16384`, a pre-trained legal domain seq-to-seq transformer model capable of handling long documents up to 16,384 input tokens.
* **Dataset**: The Indian Legal Dataset (ILDC) from Hugging Face, which includes 7,030 training samples and 1,000 testing samples with high-quality reference summaries from legal professionals.
* **PEFT/LoRA Implementation**: To improve training efficiency, LoRA was used with a rank of 8, alpha of 16, and a dropout of 0.1, targeting the attention and feedforward layers of the model.

* **Reward Function**: The RL process is guided by a reward function calculated as a weighted average of three scores:
    * **Entailment Score (E)**: Measures how much the generated summary is supported by the original text, calculated using cosine similarity of their embeddings.
    * **Kullback-Leibler (KL) Divergence (KL)**: Ensures the fine-tuned model does not stray too far from the base model's language patterns.
    * **Length Penalty (L)**: Discourages the generation of overly short summaries.
    The net reward is formulated as: `Net Reward = 0.6*E + 0.3*KL + 0.1*L`.

The overall methodology for the summarization system is illustrated below:

[Methodology for Fine-Tuning Legal-LED with LoRA + RL]
<img width="1481" height="571" alt="image" src="https://github.com/user-attachments/assets/ebb1ac6f-6f68-4db6-938e-e2c60535a1f5" />


### System II: Legal Question Answering

A Retrieval-Augmented Generation (RAG) approach is used for the question-answering system.

* **System Architecture**:
    * **Vector Database**: Pinecone is used for efficient storage and retrieval of document embeddings.
    * **Embedding Generation**: The `multilingual-e5-large` model generates vector representations for documents and user queries.
    * **Language Model**: Groq's LLaMA3-70b model is used to generate answers based on the retrieved contexts.
* **Document Processing**: Legal texts are chunked using a sentence-preserving method, with each chunk being a maximum of 505 tokens. These chunks are then converted to embeddings and stored in Pinecone.
* **Query Processing**: The system supports two workflows:
    1.  **Regular Questions**: For general legal questions, the system checks a cache for similar past queries before searching the entire legal knowledge base.
    2.  **Questions on New Documents**: When a user uploads a new document, it is processed and stored in a dedicated namespace for targeted question answering.

The question-answering workflows are depicted in the following diagram:

[Methodology of Legal Question Answering]
<img width="2100" height="1092" alt="image" src="https://github.com/user-attachments/assets/f5140694-328f-424d-abab-8e26a612e542" />


### Web-Based Frontend Tool

A user-friendly web interface was developed using FastAPI for the backend and NextJS for the frontend, allowing users to upload documents, request summaries, and ask questions in real-time.

[Frontend Interface for Legal Summary - 1]
<img width="837" height="644" alt="image" src="https://github.com/user-attachments/assets/41a1e941-1104-434b-964d-01549b3d1c17" />
[Frontend Interface for Legal Summary - 2]
<img width="1896" height="969" alt="image" src="https://github.com/user-attachments/assets/9304f89f-4c9c-4503-a0c1-630a12e5a87a" />


## Results and Analysis

### System I: Legal Document Summarization

The fine-tuned model showed significant improvements over the base model in summarizing legal texts.

* **Performance Metrics**: The LoRA-fine-tuned model demonstrated superior performance across various metrics, including entailment, factual accuracy, and completeness. The optimal performance was achieved with a LoRA rank of 8 and a KL divergence weight of 0.3.


* **Entailment and Summary Length**: The fine-tuned model consistently achieved higher entailment scores while maintaining a more appropriate summary length compared to the base model.

    [Steps vs Entailment]
<img width="682" height="482" alt="image" src="https://github.com/user-attachments/assets/2b471d3e-8e6e-4f0d-8bdf-5e6dd74f3071" />


    [Steps vs Summary Length]
<img width="685" height="483" alt="image" src="https://github.com/user-attachments/assets/3d78a123-0bb3-4786-92b0-8b5bdd71bee7" />


* **Efficiency**: Using LoRA reduced the number of trainable parameters to just 0.8% of the original model, decreasing the training memory requirement from 24.8 GB to 9.8 GB and increasing the training speed by nearly four times.

    [LoRA Memory and Speed Comparison]
<img width="1182" height="486" alt="image" src="https://github.com/user-attachments/assets/e6c1f23d-d900-4c34-bad4-5562ffd9c668" />


* **Performance on Different Document Lengths**: The fine-tuned model showed increasingly better performance over the base model as the input document length increased, with a 31.2% improvement for documents over 4,000 tokens. 

    [Performance by Document Length]
<img width="1182" height="486" alt="image" src="https://github.com/user-attachments/assets/85ce6f93-c0a1-423f-9f20-b63710aee4af" />


* **Hyperparameter Tuning**: A heatmap analysis of different LoRA ranks and KL divergence weights confirmed that a rank of 8 and a KL weight of 0.3 provided the best performance.

    [KL divergence weight against LoRA Rank]
<img width="742" height="544" alt="image" src="https://github.com/user-attachments/assets/94dd7ab1-f506-4c86-8ecc-f56be3583ca3" />


### System II: Legal Question Answering

The RAG system was evaluated on a dataset of 1,897 unique question-answer pairs from approximately 300 Indian legal cases.

* **Evaluation Metrics**:
    * **ROUGE Scores**: The system achieved average ROUGE-1, ROUGE-2, and ROUGE-L scores of 0.3, 0.18, and 0.27, respectively. The scores were impacted by the LLM generating longer answers than the ground truth. 
        [Average ROUGE Scores]
<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/500b64b4-494b-4b28-847e-2d5e77bfb8da" />

    * **BLEU Scores**: The BLEU scores varied, with some examples showing high scores (above 0.6) while most were moderate to low.
        [Individual BLEU Scores]
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/e0bced1f-acf2-4c56-bc5e-226581e21f77" />

    * **Semantic Similarity**: The average cosine similarity score between generated and ground-truth answers was approximately 0.4. 
        [Answer Relevancy Scores]
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/7ba95349-9f7a-40f8-9592-aab5228068b1" />

    * **Precision, Recall, and F1 Scores**: The average precision, recall, and F1-score were 0.32, 0.36, and 0.338, respectively.
        [Precision, Recall, and F1 Scores]
<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/94ef4ee9-56fa-4acc-b1c3-5bd8be2d6f8e" />

    * **System Efficiency**: The document upload time showed a linear relationship with the number of chunks, while the RAG processing time remained relatively constant at around 1.15 seconds, indicating excellent query efficiency.

## Conclusion and Future Scope

This project successfully implemented a robust application for legal text summarization and question answering. The RL-based fine-tuning significantly improved the summarization model's factual coherence and reduced hallucinations. The RAG system for question answering demonstrated efficient retrieval and generation, although the verbosity of the generated answers affected some metrics.

Future work will focus on:
* Fine-tuning the summarization transformer for a larger number of epochs, provided more compute power is available.
* Improving the question-answering module by using user-specific database indices to enhance retrieval.
* [cite_start]Fine-tuning the LLaMA 3 model specifically for legal question answering to improve accuracy and conciseness. [cite: 441, 443]
