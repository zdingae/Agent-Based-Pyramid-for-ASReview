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
- **Agent Creation**: Each layer subdivided into micro-agents, with each agent managing `docs_per_agent documents` (configurable, default: 50). This parameter allows fine-tuning the granularity of agent control:
  - Smaller `docs_per_agent` (e.g., 10-20): More agents with finer control, better adaptability but higher computational overhead
  - Larger `docs_per_agent` (e.g., 100-200): Fewer agents with coarser control, faster processing but potentially less responsive to local document clusters
- **Initial State**: All agents start with `election_score = 0.5`

#### Agent Selection & Update
- **Selection**: Agent with highest election score reviews next batch
- **Batch Strategy**: The system selects the agent with the highest election score, then reviews up to 10 documents (or all remaining if fewer) within that agent ranked by feature-based scores learned from ASReview's classifier（or manually input keywords), effectively balancing exploration through agent partitioning with exploitation through feature-based prioritization.
- **Score Update**:
  - Success rate = 0: `election_score = 0.1` (heavily penalized)
  - Success rate = 1: `election_score = 2.0` (strongly promoted)
  - Otherwise: `success_rate × (1 + remaining_ratio)`
- **Deactivation**: Agent removed when all documents reviewed

## 🔄 Mode Switching Logic
### ASReview → Agent Mode
- **Trigger Condition**: Consecutive irrelevant documents exceed `switch_threshold`
  - `switch_threshold`: Percentage of total documents (default: 0.05 = 5%)
  - Lower threshold = earlier switch to agent mode (e.g., 0.03 = 3% for aggressive switching)
  - Higher threshold = more patience with ASReview (e.g., 0.20 = 20% for conservative switching)
- **Additional Constraint**: Must find at least one relevant document beyond seeds
- **Action**: Creates pyramid structure from all remaining unreviewed documents
- **Example**: With 1000 documents and threshold=0.05, switches after 50 consecutive irrelevant findings

### Agent → ASReview Mode
- **Trigger Condition**(either): 
  - Relevant documents are rediscovered in agent mode (consecutive_irrelevant = 0)
  - All active agents complete their document reviews
- **Action**: Returns to standard ASReview active learning
- **Benefit**: Newly discovered relevant documents enrich the training set, potentially improving classifier performance
- **Flexibility**: Allows multiple mode switches during review process

## 📈 Batch Simulation Framework
The auto mode script provides a framework for running multiple simulations:
```python
# Example configuration
params = {
    'dataset_name': "Appenzeller-Herzog_2019",
    'n_simulations': 65,    # number of simulations
    'switch_threshold': 0.15,  # 15% threshold
    'n_layers': 20,
    'docs_per_agent': 10, 
    'n_important_features': 20  # number of important features extracted from RF
}
```

## 📊 Performance

Experiments on [SYNERGY](https://github.com/asreview/synergy-dataset) datasets demonstrate:
- Superior recall curves compared to standard ASReview
- Significant improvements in outlier detection
- Better coverage of heterogeneous document collections

## 🔧 Use Cases

- High-stakes medical reviews where missing critical studies has serious consequences
- Interdisciplinary research requiring diverse terminology coverage
- Comprehensive systematic reviews demanding exhaustive literature coverage

## 📚 Citation
If you use this code in your research, please cite:
```
@online{adaptive_asreview_agent,
  title={Adaptive ASReview-Agent Hybrid for Enhanced Active Learning},
  author={[Zeyu Ding]},
  year={2025},
  url={https://github.com/zdingae/Agent-Based-Pyramid-for-ASReview},
  urldate={2025-06-13}
}
```
