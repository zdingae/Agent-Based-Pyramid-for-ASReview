# Agent-Based Pyramid (ABP) for Enhanced Systematic Review Screening

An improved algorithm for ASReview that addresses the challenge of identifying outlier papers in systematic reviews.

## 🐼 Problem Statement

The exponential growth of academic literature creates significant challenges for systematic reviews. While ML-based active learning systems like ASReview excel at finding similar documents, they might create "filter bubbles" that miss relevant papers with diverse characteristics located in sparse feature regions.

## ✨ Solution

Agent-Based Pyramid (ABP) - A dynamic framework that complements ASReview with systematic exploration capabilities to ensure comprehensive literature coverage.

## 📋 Requirements
```
pip install asreview>=2.0
pip install synergy-dataset
pip install numpy pandas scikit-learn matplotlib tqdm
```
## 🎯 Key Definitions

- **Active Learning**: Machine learning approach where the model interactively queries for labels on the most informative samples
- **Agent-based Pyramid**: Hierarchical structure organizing unreviewed documents into layers based on outlier scores
- **Mode Switching**: Adaptive mechanism to transition between ASReview and agent-based selection
- **Feature Importance**: Weights extracted from Random Forest classifier to identify discriminative terms
- **Election Score**: Agent performance metric balancing success rate and remaining document potential

## 🦒 Metrics

### Raw Metrics
- **Documents Reviewed**: Total number of documents labeled
- **Relevant Found**: Cumulative count of relevant documents discovered

### Adjusted Metrics
- **Adjusted Relevant Count**: Relevant documents found excluding seed documents
- **Recall**: Proportion of relevant documents found (excluding seeds)
- **Precision**: Ratio of relevant to total reviewed documents

### Performance Metrics
- **Rolling Efficiency**: Success rate over the last 50 reviewed documents
- **Consecutive Irrelevant**: Count of consecutive non-relevant documents (triggers mode switch)
- **Mode Distribution**: Time spent in ASReview vs Agent mode

## 🚀 Phase 1: Active Learning in ASReview
### Model Architecture
- **Classifier**: Random Forest with 100 estimators
  - `max_features='sqrt'` for reduced overfitting
  - Provides feature importance scores for Phase 2
- **Feature Extractor**: TF-IDF Vectorizer
  - Unigrams and bigrams (`ngram_range=(1,2)`)
  - Top 1000 features
  - English stop words removed
- **Query Strategy**:Maximum relevance prediction (selects documents with highest predicted probability)

### Training Data & Prior Knowledge
- **Seed Selection**: 2 documents (1 relevant, 1 irrelevant when possible)
- **Stratified Sampling**: Ensures initial training set contains both classes
- **No External Knowledge**: Pure active learning from document text

### Class Balancing
- **Balanced Sampler**:  Maintains 1:1 ratio between relevant/irrelevant in training
- **Dynamic Resampling**: Adjusts for class imbalance during each iteration
- **Fallback Strategy**: Ensures both classes present even with extreme imbalance

### Switching Condition
- **Threshold**: Default 5% of total documents (configurable)
- **Trigger**: Consecutive irrelevant documents exceed threshold
- **Constraint**: Must find at least one relevant document beyond seeds before switching
- **Safety**: Resets counter at 2× threshold if no relevant found


## 🚀 Phase 2: Agent-Based Pyramid Method
### Part 1: Distance-Based Scoring (40% weight)
- **Similarity Calculation**: Cosine similarity between unreviewed and reviewed documents
- **Outlier Detection**: Documents with low average similarity to reviewed set
- **Rationale**: Targets potentially relevant documents in unexplored regions. Since highly similar relevant documents are likely already discovered by ASReview's active learning, outliers (low similarity documents) may contain relevant documents that use different terminology, perspectives, or methodologies - exactly the "hidden gems" that traditional active learning might miss

### Part 2: Feature-Based Scoring (60% weight)
#### Mode 1 (Manual): User-Defined Keywords
```python
# Example for Wilson's disease dataset
keywords = ['Wilson', 'copper', 'hepatic', 'treatment', 'patient', 
            'penicillamine', 'zinc']
```
- **Scoring**: Keyword frequency normalized by document length
- **Weighting**: Square root transformation to reduce dominance of high-frequency terms

#### Mode 2 (Auto): Predicted Words of Importance (Using Random Forest for current version)
- **Source**: Top features from Random Forest `feature_importances_`
- **Dynamic Updates**: Re-extracted after each ASReview iteration
- **Ranking**: Features weighted by importance scores (1/rank weighting)
- **Adaptability**: Learns domain-specific terms without manual input

#### Pyramid Construction
- **Score Calculation**: Combined distance + feature scores for all unreviewed documents
- **Stratification**: Documents sorted by score and divided into `n_layers` (default: 5)
- **Agent Creation**: Each layer subdivided into micro-agents, with each agent managing `docs_per_agent documents` (configurable, default: 50)
- **Initial State**: All agents start with `election_score = 0.5`

## 📊 Performance

Experiments on [SYNERGY](https://github.com/asreview/synergy-dataset) datasets demonstrate:
- Superior recall curves compared to standard ASReview
- Significant improvements in outlier detection
- Better coverage of heterogeneous document collections

## 🔧 Use Cases

- High-stakes medical reviews where missing critical studies has serious consequences
- Interdisciplinary research requiring diverse terminology coverage
- Comprehensive systematic reviews demanding exhaustive literature coverage
