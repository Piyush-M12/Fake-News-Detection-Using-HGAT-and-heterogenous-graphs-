# Temporal Fake-News-Detection-Using-HGAT-and-heterogenous-graphs-


# 📰 Temporal Evolution-Aware Fake News Detection using HGAT

## 📌 Overview

Fake news spreads rapidly across digital platforms, often reaching millions before being corrected. Traditional detection methods rely heavily on textual content, making them vulnerable to well-written misinformation.

This project tackles the problem using **graph-based deep learning + temporal modeling**.

We propose a system that:

* Models relationships between news, creators, and topics using a **Heterogeneous Information Network (HIN)**
* Applies **Hierarchical Graph Attention Networks (HGAT)** for structural learning
* Extends it with **Temporal-HGAT (HGAT + LSTM)** to capture how news credibility evolves over time

👉 Core idea: **Fake news is dynamic — not static.**

---

## 🎯 Problem Statement

* Fake news spreads faster than fact-checking
* Text-only models fail on sophisticated misinformation
* Existing graph models ignore **temporal evolution**
* Class imbalance (especially in real-world datasets) affects performance

**Research Question:**

> How can we model the evolving nature of news using graph structures and time-aware learning to improve fake news detection?

---

## 🚀 Key Contributions

* Built a **Heterogeneous Information Network (HIN)** from raw datasets
* Implemented **HGAT with node-level and schema-level attention**
* Designed a **Temporal-HGAT framework using LSTM**
* Ensured **leakage-free training** (no data contamination across splits)
* Addressed **class imbalance** using:

  * Weighted loss
  * Improved training strategies
* Performed experiments across **multiple train splits (20–80%)**

---

## 📂 Datasets

### 🏛 PolitiFact

* Domain: Politics & Policy
* Size: ~900 articles
* Balanced dataset (real ≈ fake)
* Strong signal from publisher domains

### 📰 GossipCop

* Domain: Entertainment & Celebrity
* Size: ~22,000 articles
* Highly imbalanced (~10:1 real:fake)
* Harder to detect fake news due to imbalance

---

## 🧠 Methodology

### 1️⃣ Heterogeneous Information Network (HIN)

We construct a graph with:

#### 🔹 Node Types

* **News Articles** → TF-IDF features (3000 dims)
* **Creators** → URL domain-based features (500 dims)
* **Subjects** → Keyword-based features (100 dims)

#### 🔹 Edge Types

* Creator → Article (publishes)
* Article → Subject (topic_of)

This allows the model to capture **structural relationships**, not just text.

---

### 2️⃣ HGAT (Hierarchical Graph Attention Network)

HGAT learns representations using two levels of attention:

#### 🔸 Node-Level Attention

* Aggregates information from neighbors
* Learns importance of:

  * Creators
  * Subjects

#### 🔸 Schema-Level Attention

* Combines different node types
* Learns which source (creator vs subject) is more important

👉 Output: A rich embedding for each news article

---

### 3️⃣ Temporal-HGAT (Main Innovation)

To capture **time dynamics**, we extend HGAT:

#### Step 1: Temporal Partitioning

* Split dataset into:

  * Early
  * Middle
  * Late
* Based on timestamps (or proxy ordering)
* **Leakage-free**: boundaries computed using only training data

#### Step 2: Per-Snapshot HGAT

* Train separate HGAT models for each time phase
* Each produces an embedding

#### Step 3: LSTM Sequencing

* Combine embeddings as a sequence:

```text
Early → Middle → Late
```

* Pass into LSTM → final classifier

👉 Captures how patterns evolve over time

* Fake news → unstable patterns
* Real news → consistent patterns

---

## ⚙️ Installation

```bash
pip install numpy pandas scikit-learn torch tqdm matplotlib
```

---


```

### 2. Add Dataset Files

Place these CSV files in the root folder:

* politifact_real.csv
* politifact_fake.csv
* gossipcop_real.csv
* gossipcop_fake.csv



## 📊 Experiments & Results

### 📌 Static HGAT

* Performs well on balanced datasets
* Struggles with imbalanced data (GossipCop)

### 📌 Temporal-HGAT

* Improves performance across all splits
* Significant boost in **F1-score (fake detection)**

#### Key Insight:

* Accuracy alone is misleading
* Temporal modeling improves **recall of fake news**

---

## 📈 Observations

* URL domains are strong credibility indicators
* Class imbalance severely impacts F1 score
* Temporal context helps recover fake detection performance
* Different datasets rely on different time phases:

  * PolitiFact → Late phase important
  * GossipCop → Early phase important

---

## ⚠️ Limitations

* Uses TF-IDF (no deep semantic understanding)
* Creator = URL domain (simplified representation)
* No user interaction or propagation modeling
* Temporal modeling is approximate (not true tracking)
* Computationally expensive (multiple HGAT models)

---

## 🔮 Future Work

* Replace TF-IDF with **BERT / RoBERTa embeddings**
* Add **user behavior & social network data**
* Model **news propagation (shares, retweets)**
* Use **dynamic graph neural networks**
* Build **real-time fake news detection system**

---

## 📁 Project Structure

```text
├── data/
├── models/
├── results/
├── main.py
├── README.md
```

---

## 👥 Team

* Hridey Agarwal
* Piyush Mahajan
* Ridhima Garg

---

## 📌 Conclusion

* Graph-based approaches outperform text-only models
* Temporal modeling is critical for fake news detection
* Even simple features (like URL domains) provide strong signals
* Temporal-HGAT shows consistent improvement across datasets


