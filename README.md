Automated Performance Optimization of RAG-based Chatbots Using Reinforcement Learning

Author: Md Kabir
Program: Doctor of Engineering in Artificial Intelligence and Machine Learning
Institution: The George Washington University

Overview

This project contains the full experimental code and evaluation framework developed as part of the doctoral research on optimizing Retrieval-Augmented Generation (RAG) chatbots using Reinforcement Learning (RL).

The objective of this work is to automate performance tuning in RAG-based question answering chatbots by dynamically learning the best retrieval configuration through reinforcement signals. This reduces dependence on human annotations and enables continuous, automated model calibration.

Main Components

Baseline RAG Pipeline

Implements BM25 and DPR retrievers

Supports hybrid retrieval with alpha-weighted combination

Includes sentence-level chunking and filtering

Uses GPT-based generation through the OpenAI API

Evaluation Metrics (RAGAS-aligned)

Faithfulness: checks factual consistency between answer and context

Answer Relevance: measures semantic similarity between question and generated answer

Context Relevance: measures retrieval completeness and contextual coverage

Optimization Models

Global Policy: Shared Reward Network (SRN-27), Contextual Bandit (CB-27)

Local Policies: Δ-MoveNet RF Direct Method and RF Hybrid LinUCB

Each learns to select the optimal retrieval parameters (alpha, top-k, and chunk size) that maximize the combined reward.

Methodological Contribution

Introduces a continuous feedback loop between RAG output evaluation and retrieval selection.

Demonstrates that RL and contextual bandit-based policies outperform static retrieval configurations.

Provides an automated evaluation method that does not rely on human annotation.

Experimental Setup

Dataset: Modified RAGAS / WikiEval dataset
Retrieval parameters:
alpha ∈ {0.3, 0.5, 0.7}
top-k ∈ {1, 3, 5}
chunk size ∈ {50, 100, 150}

Reward Function:
R = (Faithfulness + Answer Relevance + Context Relevance) / 3

Algorithms used: Linear UCB, Random Forest, Contextual Bandit, and LinUCB-based Δ-MoveNet
Models used: OpenAI GPT-4o-mini for generation and evaluation

Requirements

Install dependencies:
pip install torch sentence-transformers rank-bm25 faiss-cpu langchain openai nltk pandas scikit-learn gym

Results Summary

RL-based RAG models achieved higher mean reward and lower regret than static baselines.

Δ-MoveNet with LinUCB showed adaptive improvement in limited exploration settings.

The framework supports annotation-free, automated, and scalable evaluation for RAG systems.
results/ - Contains experiment outputs and Excel summaries
utils/ - Helper functions for metrics and preprocessing

Citation

Kabir, M. (2025). Automated Performance Optimization of RAG-based Chatbots Using Reinforcement Learning.
Doctor of Engineering Thesis, The George Washington University.
