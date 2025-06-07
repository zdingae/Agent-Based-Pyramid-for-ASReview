# Agent-Based Pyramid (ABP) for Enhanced Systematic Review Screening

An improved algorithm for ASReview that addresses the challenge of identifying outlier papers in systematic reviews.

## 🎯 Problem Statement

The exponential growth of academic literature creates significant challenges for systematic reviews. While ML-based active learning systems like ASReview excel at finding similar documents, they often create "filter bubbles" that miss relevant papers with diverse characteristics located in sparse feature regions.

## 🚀 Solution

Agent-Based Pyramid (ABP) - A dynamic framework that complements ASReview with systematic exploration capabilities to ensure comprehensive literature coverage.

## ✨ Key Features

### Dual-Mode Operation
- **Manual Mode**: Researchers input domain-specific keywords for targeted exploration
- **Automated Mode**: Automatically extracts important features from ASReview's Random Forest classifier

### Adaptive Performance
- Real-time performance monitoring
- Automatic switching to agent-based exploration when blind spots are detected
- Hierarchical document organization based on outlier scores

### Smart Prioritization
- Election-based document prioritization
- Flexible switching between ML and agent-based approaches
- Enhanced recall for hard-to-find papers

## 📊 Performance

Experiments on SYNERGY datasets demonstrate:
- Superior recall curves compared to standard ASReview
- Significant improvements in outlier detection
- Better coverage of heterogeneous document collections

## 🔧 Use Cases

- High-stakes medical reviews where missing critical studies has serious consequences
- Interdisciplinary research requiring diverse terminology coverage
- Comprehensive systematic reviews demanding exhaustive literature coverage