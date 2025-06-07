import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from tqdm import tqdm
import warnings
import os
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# ASReview 2.0 imports
import asreview as asr
from asreview.models.classifiers import RandomForest
from asreview.models.feature_extractors import Tfidf
from asreview.models.balancers import Balanced
from asreview.models.queriers import Max

# Suppress warnings
warnings.filterwarnings('ignore')


class AdaptiveASReviewAgentHybrid:
    """
    Adaptive hybrid approach using ASReview's built-in functionality:
    1. Start with ASReview's RandomForest classifier
    2. Extract important features from feature_importances_
    3. Switch to agent-based pyramid when hitting sparse regions
    4. Use important features instead of manual keywords
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 labels: pd.Series,
                 switch_threshold: float = 0.05,  # 5% consecutive irrelevant
                 n_layers: int = 5,
                 docs_per_agent: int = 50,
                 initial_seed_size: int = 2,
                 n_important_features: int = 15):  # Number of features to use for scoring
        
        self.data = data
        self.labels = labels
        self.switch_threshold = switch_threshold
        self.n_layers = n_layers
        self.docs_per_agent = docs_per_agent
        self.initial_seed_size = initial_seed_size
        self.n_important_features = n_important_features
        
        # Prepare texts
        self.texts = self._prepare_texts()
        
        # Global tracking
        self.reviewed = set()
        self.relevant_found = set()
        self.review_history = []
        
        # Track seed documents separately for recall adjustment
        self.seed_documents = []
        self.relevant_in_seed = 0
        
        # ASReview components
        self.feature_extractor = None
        self.classifier = None
        self.balancer = None
        self.querier = None
        self.feature_matrix = None
        self.vocabulary = None
        self.feature_names = None
        
        # Important features tracking
        self.important_features = []
        self.feature_importance_history = []
        
        # Agent components
        self.agents = None
        self.agent_mode = False
        
        # Metrics - now with adjusted recall
        self.metrics = {
            'n_reviewed': [],
            'n_relevant_found': [],
            'n_relevant_found_adjusted': [],  # New: adjusted for seed documents
            'recall': [],  # New: actual recall excluding seed
            'mode': [],
            'precision': [],
            'efficiency': []
        }
        
        # Performance tracking
        self.consecutive_irrelevant = 0
        self.mode_switches = []
        
    def _prepare_texts(self) -> List[str]:
        """Prepare texts by combining title and abstract"""
        texts = []
        for idx, row in self.data.iterrows():
            title = str(row.get('title', ''))
            abstract = str(row.get('abstract', ''))
            combined = f"{title} {abstract}".strip()
            texts.append(combined if combined else "empty document")
        return texts
    
    def _initialize_asreview(self):
        """Initialize ASReview components"""
        # Initialize ASReview models
        self.feature_extractor = Tfidf(
            columns=['text'],
            ngram_range=(1, 2),
            stop_words='english',
            max_features=1000
        )
        
        # Use ASReview's RandomForest classifier
        self.classifier = RandomForest(
            n_estimators=100,
            max_features='sqrt',
            random_state=42
        )
        
        # Balance strategy
        self.balancer = Balanced(ratio=1.0)
        
        # Query strategy
        self.querier = Max()
        
        # Fit feature extraction on all texts
        # Create a DataFrame for feature extraction
        text_df = pd.DataFrame({'text': self.texts})
        self.feature_matrix = self.feature_extractor.fit_transform(text_df)
        
        # Extract feature names from the pipeline
        self._extract_feature_names_from_pipeline()
        
        # Initialize with seed documents
        unreviewed = list(range(len(self.texts)))
        
        # Try to get stratified seeds
        pos_indices = [i for i in unreviewed if self.labels.iloc[i] == 1]
        neg_indices = [i for i in unreviewed if self.labels.iloc[i] == 0]
        
        seeds = []
        if pos_indices and neg_indices:
            seeds.append(np.random.choice(pos_indices))
            seeds.append(np.random.choice(neg_indices))
        else:
            seeds = np.random.choice(unreviewed, 
                                   min(self.initial_seed_size, len(unreviewed)), 
                                   replace=False).tolist()
        
        # Store seed documents
        self.seed_documents = seeds
        
        # Review seeds
        for doc_idx in seeds:
            self._review_document(doc_idx, mode='asreview_init')
        
        # Count relevant documents in seed
        self.relevant_in_seed = len([d for d in self.seed_documents if d in self.relevant_found])
    
    def _extract_feature_names_from_pipeline(self):
        """Extract feature names from ASReview's pipeline"""
        try:
            # ASReview's feature extractor might be a pipeline
            if hasattr(self.feature_extractor, 'steps'):
                # It's a pipeline, find the vectorizer step
                for step_name, step_obj in self.feature_extractor.steps:
                    if hasattr(step_obj, 'get_feature_names_out'):
                        self.feature_names = step_obj.get_feature_names_out()
                        break
                    elif hasattr(step_obj, 'get_feature_names'):
                        self.feature_names = step_obj.get_feature_names()
                        break
            else:
                # Direct vectorizer
                if hasattr(self.feature_extractor, 'get_feature_names_out'):
                    self.feature_names = self.feature_extractor.get_feature_names_out()
                elif hasattr(self.feature_extractor, 'get_feature_names'):
                    self.feature_names = self.feature_extractor.get_feature_names()
            
            # If still no feature names, try to access the underlying vectorizer
            if self.feature_names is None and hasattr(self.feature_extractor, 'model'):
                model = self.feature_extractor.model
                if hasattr(model, 'get_feature_names_out'):
                    self.feature_names = model.get_feature_names_out()
                elif hasattr(model, 'get_feature_names'):
                    self.feature_names = model.get_feature_names()
                    
        except Exception as e:
            # Create dummy feature names
            if self.feature_matrix is not None:
                n_features = self.feature_matrix.shape[1]
                self.feature_names = [f"feature_{i}" for i in range(n_features)]
    
    def _extract_important_features(self):
        """Extract important features from RandomForest classifier"""
        # Get feature importances directly from the classifier
        importances = self.classifier.feature_importances_
        
        if importances is None or len(importances) == 0:
            return
        
        # Get indices of top features
        top_indices = np.argsort(importances)[-self.n_important_features:][::-1]
        
        # Map indices to actual feature names
        feature_names = []
        
        if self.feature_names is not None:
            for idx in top_indices:
                if idx < len(self.feature_names):
                    feature_name = self.feature_names[idx]
                    importance_score = importances[idx]
                    feature_names.append((feature_name, importance_score))
        else:
            # Fallback: use feature indices
            for idx in top_indices:
                feature_name = f"feature_{idx}"
                importance_score = importances[idx]
                feature_names.append((feature_name, importance_score))
        
        # Update important features
        self.important_features = [name for name, _ in feature_names]
        
        # Record history
        self.feature_importance_history.append({
            'n_reviewed': len(self.reviewed),
            'n_relevant': len(self.relevant_found),
            'top_features': feature_names  # Store all extracted features
        })
    
    def _review_document(self, doc_idx: int, mode: str):
        """Review a single document and update tracking"""
        if doc_idx in self.reviewed:
            return False
        
        self.reviewed.add(doc_idx)
        self.review_history.append(doc_idx)
        
        is_relevant = self.labels.iloc[doc_idx] == 1
        if is_relevant:
            self.relevant_found.add(doc_idx)
            self.consecutive_irrelevant = 0
        else:
            self.consecutive_irrelevant += 1
        
        # Update metrics with both raw and adjusted counts
        self.metrics['n_reviewed'].append(len(self.reviewed))
        self.metrics['n_relevant_found'].append(len(self.relevant_found))
        
        # Calculate adjusted relevant count (excluding seed relevant)
        adjusted_relevant = len(self.relevant_found) - self.relevant_in_seed
        self.metrics['n_relevant_found_adjusted'].append(adjusted_relevant)
        
        # Calculate actual recall (excluding seed documents)
        total_relevant_excluding_seed = self.labels.sum() - self.relevant_in_seed
        if total_relevant_excluding_seed > 0:
            recall = adjusted_relevant / total_relevant_excluding_seed
        else:
            recall = 0.0
        self.metrics['recall'].append(recall)
        
        self.metrics['mode'].append(mode)
        
        if len(self.reviewed) > 0:
            self.metrics['precision'].append(len(self.relevant_found) / len(self.reviewed))
        else:
            self.metrics['precision'].append(0)
        
        # Calculate rolling efficiency (last 50 documents)
        recent_start = max(0, len(self.review_history) - 50)
        recent_docs = self.review_history[recent_start:]
        recent_relevant = sum(1 for d in recent_docs if d in self.relevant_found)
        self.metrics['efficiency'].append(recent_relevant / len(recent_docs) if recent_docs else 0)
        
        return is_relevant
    
    def _run_asreview_iteration(self, batch_size: int = 10) -> int:
        """Run one iteration of ASReview active learning"""
        unreviewed = [i for i in range(len(self.texts)) if i not in self.reviewed]
        
        if not unreviewed:
            return 0
        
        # Get training data
        reviewed_list = list(self.reviewed)
        reviewed_labels = np.array([1 if i in self.relevant_found else 0 
                                   for i in reviewed_list])
        
        if len(set(reviewed_labels)) >= 2:
            # Get features for training
            X_train = self.feature_matrix[reviewed_list]
            
            # Apply balancing
            # Compute sample weights
            sample_weights = self.balancer.compute_sample_weight(reviewed_labels)
            
            # Use weights to resample the training data
            n_samples = len(reviewed_labels)
            
            # Ensure we have valid sample weights
            if sample_weights.sum() > 0:
                sample_indices = np.random.choice(
                    n_samples, 
                    size=n_samples, 
                    replace=True, 
                    p=sample_weights/sample_weights.sum()
                )
            else:
                # Fallback to uniform sampling if weights are invalid
                sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            X_balanced = X_train[sample_indices]
            y_balanced = reviewed_labels[sample_indices]
            
            # Ensure balanced data has both classes
            if len(np.unique(y_balanced)) < 2:
                # Force include both classes if balancing removed one
                pos_idx = np.where(reviewed_labels == 1)[0]
                neg_idx = np.where(reviewed_labels == 0)[0]
                
                if len(pos_idx) > 0 and len(neg_idx) > 0:
                    # Create a balanced set with both classes
                    min_class_size = min(len(pos_idx), len(neg_idx))
                    balanced_indices = np.concatenate([
                        np.random.choice(pos_idx, min_class_size, replace=True),
                        np.random.choice(neg_idx, min_class_size, replace=True)
                    ])
                    np.random.shuffle(balanced_indices)
                    
                    X_balanced = X_train[balanced_indices]
                    y_balanced = reviewed_labels[balanced_indices]
            
            # Train classifier
            self.classifier.fit(X_balanced, y_balanced)
            
            # Extract important features after training
            self._extract_important_features()
            
            # Predict on unreviewed
            X_unreviewed = self.feature_matrix[unreviewed]
            
            # Get predictions/probabilities
            pred_proba = self.classifier.predict_proba(X_unreviewed)
            
            # Handle case where classifier might only predict one class
            if pred_proba.shape[1] == 1:
                # If only one class, use random scores with slight preference
                scores = np.random.rand(len(unreviewed)) * 0.5
            else:
                scores = pred_proba[:, 1]
            
            # Select top scoring documents
            n_select = min(batch_size, len(unreviewed))
            top_indices = np.argsort(scores)[-n_select:][::-1]
            selected = [unreviewed[i] for i in top_indices]
        else:
            # Random selection if not enough diversity
            n_select = min(batch_size, len(unreviewed))
            selected = np.random.choice(unreviewed, n_select, replace=False).tolist()
        
        # Review selected documents
        found_relevant = 0
        for doc_idx in selected:
            if self._review_document(doc_idx, mode='asreview'):
                found_relevant += 1
        
        return found_relevant
    
    def _calculate_feature_based_scores(self, indices: List[int]) -> np.ndarray:
        """Calculate document scores based on important features"""
        scores = np.zeros(len(indices))
        
        # If we have feature names that start with "feature_", use feature matrix directly
        if (self.important_features and 
            all(f.startswith('feature_') for f in self.important_features[:min(5, len(self.important_features))])):
            # Use feature matrix directly
            for i, doc_idx in enumerate(indices):
                feature_score = 0
                for j, feature in enumerate(self.important_features):
                    try:
                        # Extract feature index
                        feat_idx = int(feature.split('_')[1])
                        if feat_idx < self.feature_matrix.shape[1]:
                            weight = 1.0 / (j + 1)
                            feature_value = self.feature_matrix[doc_idx, feat_idx]
                            feature_score += weight * feature_value
                    except:
                        pass
                scores[i] = feature_score
        else:
            # Text-based scoring
            features_to_use = self.important_features  # Use all extracted features
            
            for i, doc_idx in enumerate(indices):
                text = self.texts[doc_idx].lower()
                text_length = len(text.split())
                
                if text_length > 0:
                    feature_score = 0
                    for j, feature in enumerate(features_to_use):
                        if feature.lower() in text:
                            # Higher weight for more important features
                            weight = 1.0 / (j + 1)
                            count = text.count(feature.lower())
                            feature_score += weight * np.sqrt(count)
                    
                    scores[i] = feature_score / np.sqrt(text_length)
        
        return scores
    
    def _calculate_outlier_scores(self, indices: List[int]) -> np.ndarray:
        """Calculate outlier scores for documents"""
        scores = np.zeros(len(indices))
        
        # Method 1: Distance from reviewed documents
        if len(self.reviewed) > 0 and self.feature_matrix is not None:
            reviewed_features = self.feature_matrix[list(self.reviewed)]
            unreviewed_features = self.feature_matrix[indices]
            
            # Calculate average similarity to reviewed docs
            similarities = cosine_similarity(unreviewed_features, reviewed_features)
            avg_similarity = similarities.mean(axis=1)
            
            # Lower similarity = higher outlier score
            distance_scores = 1 - avg_similarity
            scores += distance_scores * 0.4
        
        # Method 2: Feature-based scoring using important features
        feature_scores = self._calculate_feature_based_scores(indices)
        
        # Normalize feature scores
        if feature_scores.max() > 0:
            feature_scores = feature_scores / feature_scores.max()
        
        scores += feature_scores * 0.6
        
        return scores
    
    def _create_agent_pyramid(self):
        """Create pyramid structure with agents for outlier detection"""
        # Get unreviewed documents
        unreviewed_indices = [i for i in range(len(self.texts)) if i not in self.reviewed]
        
        if not unreviewed_indices:
            return
        
        # Calculate scores for unreviewed documents
        scores = self._calculate_outlier_scores(unreviewed_indices)
        
        # Sort by score
        sorted_pairs = sorted(zip(unreviewed_indices, scores), 
                            key=lambda x: x[1], reverse=True)
        sorted_indices = [idx for idx, _ in sorted_pairs]
        
        # Create layers
        layer_size = len(sorted_indices) // self.n_layers
        self.agents = []
        
        for layer_id in range(self.n_layers):
            start_idx = layer_id * layer_size
            if layer_id == self.n_layers - 1:
                layer_docs = sorted_indices[start_idx:]
            else:
                layer_docs = sorted_indices[start_idx:start_idx + layer_size]
            
            # Create micro-agents within layer
            n_agents_in_layer = max(1, len(layer_docs) // self.docs_per_agent)
            agent_size = len(layer_docs) // n_agents_in_layer
            
            for agent_idx in range(n_agents_in_layer):
                agent_start = agent_idx * agent_size
                if agent_idx == n_agents_in_layer - 1:
                    agent_docs = layer_docs[agent_start:]
                else:
                    agent_docs = layer_docs[agent_start:agent_start + agent_size]
                
                agent = {
                    'id': f"L{layer_id}_A{agent_idx}",
                    'layer': layer_id,
                    'documents': agent_docs,
                    'reviewed': set(),
                    'relevant_found': set(),
                    'election_score': 0.5,
                    'is_active': True
                }
                
                self.agents.append(agent)
    
    def _run_agent_iteration(self) -> int:
        """Run one iteration of agent-based selection"""
        if not self.agents:
            return 0
        
        # Find agent with highest election score
        active_agents = [a for a in self.agents if a['is_active']]
        if not active_agents:
            return 0
        
        # Sort by election score
        active_agents.sort(key=lambda x: x['election_score'], reverse=True)
        selected_agent = active_agents[0]
        
        # Review documents from selected agent
        agent_docs = [d for d in selected_agent['documents'] 
                     if d not in selected_agent['reviewed']]
        
        if not agent_docs:
            selected_agent['is_active'] = False
            return 0
        
        # Score documents within agent using important features
        doc_scores = [(d, self._calculate_feature_based_scores([d])[0]) 
                     for d in agent_docs]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Review top scoring documents
        batch_size = min(10, len(agent_docs))
        batch = [d for d, _ in doc_scores[:batch_size]]
        
        found_relevant = 0
        for doc_idx in batch:
            selected_agent['reviewed'].add(doc_idx)
            if self._review_document(doc_idx, mode='agent'):
                selected_agent['relevant_found'].add(doc_idx)
                found_relevant += 1
        
        # Update agent election score
        if len(selected_agent['reviewed']) > 0:
            success_rate = len(selected_agent['relevant_found']) / len(selected_agent['reviewed'])
            remaining_ratio = 1 - len(selected_agent['reviewed']) / len(selected_agent['documents'])
            
            # Emphasize extreme values
            if success_rate == 0:
                selected_agent['election_score'] = 0.1
            elif success_rate == 1:
                selected_agent['election_score'] = 2.0
            else:
                selected_agent['election_score'] = success_rate * (1 + remaining_ratio)
        
        # Check if agent is complete
        if len(selected_agent['reviewed']) >= len(selected_agent['documents']):
            selected_agent['is_active'] = False
        
        return found_relevant
    
    def _check_mode_switch(self) -> bool:
        """Check if we should switch modes"""
        switch_threshold_docs = int(self.switch_threshold * len(self.texts))
        
        # NEW CONDITION: Must have found at least one relevant document before switching to agent mode
        # This ensures ASReview continues until it finds something to learn from
        has_found_relevant = len(self.relevant_found) > self.relevant_in_seed
        
        if not self.agent_mode and self.consecutive_irrelevant >= switch_threshold_docs:
            # Only switch to agent mode if we've found at least one relevant document
            if has_found_relevant:
                # Switch to agent mode
                self.agent_mode = True
                self.consecutive_irrelevant = 0
                self.mode_switches.append({
                    'iteration': len(self.reviewed),
                    'to_mode': 'agent',
                    'relevant_found': len(self.relevant_found)
                })
                
                # Create agent pyramid
                self._create_agent_pyramid()
                return True
            else:
                # Reset consecutive count but stay in ASReview mode
                # This prevents getting stuck if threshold is too low
                if self.consecutive_irrelevant >= switch_threshold_docs * 2:
                    self.consecutive_irrelevant = 0
        
        elif self.agent_mode and self.consecutive_irrelevant == 0:
            # Found relevant in agent mode, consider switching back
            active_agents = [a for a in self.agents if a['is_active']] if self.agents else []
            
            if not active_agents:
                self.agent_mode = False
                self.agents = None
                self.mode_switches.append({
                    'iteration': len(self.reviewed),
                    'to_mode': 'asreview',
                    'relevant_found': len(self.relevant_found)
                })
                return True
        
        return False
    
    def run(self) -> Dict:
        """Run the adaptive hybrid approach"""
        # Start with ASReview
        self._initialize_asreview()
        
        total_relevant = self.labels.sum()
        total_relevant_excluding_seed = total_relevant - self.relevant_in_seed
        iteration = 0
        
        while len(self.reviewed) < len(self.texts):
            iteration += 1
            
            # Check if we should switch modes
            self._check_mode_switch()
            
            # Run iteration based on current mode
            if self.agent_mode:
                found = self._run_agent_iteration()
                mode_str = "AGENT"
            else:
                found = self._run_asreview_iteration()
                mode_str = "ASREVIEW"
        
        return self.metrics


class BatchSimulation:
    """
    Batch simulation for adaptive hybrid approach
    Similar to Synergy simulations website
    """
    
    def __init__(self, 
                 dataset_name: str = "Appenzeller-Herzog_2019",
                 n_simulations: int = 100,
                 switch_threshold: float = 0.05,
                 n_layers: int = 5,
                 docs_per_agent: int = 50,
                 n_important_features: int = 15,
                 random_seed: int = 42):
        
        self.dataset_name = dataset_name
        self.n_simulations = n_simulations
        self.switch_threshold = switch_threshold
        self.n_layers = n_layers
        self.docs_per_agent = docs_per_agent
        self.n_important_features = n_important_features
        self.random_seed = random_seed
        
        # Results storage - now storing merged timeline
        self.merged_timeline = defaultdict(list)  # {proportion: [relevant_counts]}
        self.all_events = []  # Store all review events from all simulations
        
    def load_dataset(self):
        """Load and prepare dataset"""
        from synergy_dataset import Dataset
        
        print(f"Loading dataset: {self.dataset_name}")
        d = Dataset(self.dataset_name)
        data = d.to_frame()
        labels = data['label_included']
        
        # Filter valid documents
        valid_mask = data['title'].notna() | data['abstract'].notna()
        data = data[valid_mask].reset_index(drop=True)
        labels = labels[valid_mask].reset_index(drop=True)
        
        self.data = data
        self.labels = labels
        self.n_docs = len(data)
        self.n_relevant = labels.sum()
        
        print(f"Dataset loaded: {self.n_docs} documents, {self.n_relevant} relevant")
        print("-" * 60)
        
    def run_single_simulation(self, sim_id: int) -> Dict:
        """Run a single simulation with suppressed output"""
        # Set random seed for reproducibility
        np.random.seed(self.random_seed + sim_id)
        
        # Create a hybrid instance
        hybrid = AdaptiveASReviewAgentHybrid(
            data=self.data,
            labels=self.labels,
            switch_threshold=self.switch_threshold,
            n_layers=self.n_layers,
            docs_per_agent=self.docs_per_agent,
            initial_seed_size=2,
            n_important_features=self.n_important_features
        )
        
        # Suppress all output
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            metrics = hybrid.run()
        
        # Store review events for merging
        events = []
        for i in range(len(metrics['n_reviewed'])):
            events.append({
                'sim_id': sim_id,
                'n_reviewed': metrics['n_reviewed'][i],
                'proportion': metrics['n_reviewed'][i] / self.n_docs,
                'n_relevant_found_adjusted': metrics['n_relevant_found_adjusted'][i],
                'mode': metrics['mode'][i]
            })
        
        return {
            'sim_id': sim_id,
            'events': events,
            'seed_relevant': hybrid.relevant_in_seed
        }
    
    def run_simulations(self):
        """Run all simulations with progress bar"""
        print(f"\nRunning {self.n_simulations} simulations...")
        print(f"Parameters: threshold={self.switch_threshold}, layers={self.n_layers}, "
              f"docs_per_agent={self.docs_per_agent}, features={self.n_important_features}")
        print("-" * 60)
        
        # Progress bar
        with tqdm(total=self.n_simulations, desc="Simulations", ncols=100) as pbar:
            for sim_id in range(self.n_simulations):
                result = self.run_single_simulation(sim_id)
                
                # Add events to the global event list
                self.all_events.extend(result['events'])
                
                pbar.update(1)
        
        print("\nSimulations completed!")
        print(f"Total review events collected: {len(self.all_events)}")
        
        # Merge all events
        self._merge_timelines()
    
    def _merge_timelines(self):
        """Merge all simulation timelines"""
        print("\nMerging simulation timelines...")
        
        # Sort all events by proportion reviewed
        self.all_events.sort(key=lambda x: x['proportion'])
        
        # Create merged timeline
        current_relevant = 0
        for event in self.all_events:
            prop = event['proportion']
            # For simplicity, we'll track the cumulative count
            # In reality, each simulation contributes independently
            self.merged_timeline[prop].append(event['n_relevant_found_adjusted'])
    
    def plot_merged_recall_curve(self, save_path: str = None):
        """Plot merged recall curve with color coding"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Prepare data for plotting
        proportions = []
        relevant_counts = []
        modes = []
        
        # Sort events by proportion
        sorted_events = sorted(self.all_events, key=lambda x: x['proportion'])
        
        # Group by small proportion bins for visualization
        bin_size = 0.001  # 0.1% bins
        bins = np.arange(0, 1 + bin_size, bin_size)
        
        # Calculate mean relevant count in each bin
        binned_data = defaultdict(lambda: {'relevant': [], 'modes': []})
        
        for event in sorted_events:
            bin_idx = int(event['proportion'] / bin_size)
            if bin_idx < len(bins) - 1:
                binned_data[bins[bin_idx]]['relevant'].append(event['n_relevant_found_adjusted'])
                binned_data[bins[bin_idx]]['modes'].append(event['mode'])
        
        # Prepare data for plotting
        x_vals = []
        y_vals = []
        colors = []
        
        for bin_prop in sorted(binned_data.keys()):
            if binned_data[bin_prop]['relevant']:
                x_vals.append(bin_prop)
                # Use mean of relevant counts in this bin
                y_vals.append(np.mean(binned_data[bin_prop]['relevant']))
                
                # Determine color based on dominant mode in bin
                modes_in_bin = binned_data[bin_prop]['modes']
                asreview_count = sum(1 for m in modes_in_bin if m.startswith('asreview'))
                agent_count = sum(1 for m in modes_in_bin if m == 'agent')
                
                if agent_count > asreview_count:
                    colors.append('red')
                else:
                    colors.append('blue')
        
        # Plot as scatter plot with colors
        for i in range(len(x_vals) - 1):
            ax.plot([x_vals[i], x_vals[i+1]], [y_vals[i], y_vals[i+1]], 
                   color=colors[i], linewidth=2, alpha=0.7)
        
        # Add legend patches
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='ASReview Mode'),
            Patch(facecolor='red', label='Agent Mode')
        ]
        
        # Plot random baseline (step function)
        total_relevant_adj = self.n_relevant - 1  # Assuming 1 relevant in seed on average
        random_x = np.linspace(0, 1, 100)
        random_y = random_x * total_relevant_adj * self.n_simulations
        ax.step(random_x, random_y, 'k:', linewidth=2, where='post', 
                label='Random baseline (aggregated)', alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Proportion of Labeled Records', fontsize=14)
        ax.set_ylabel('Cumulative Relevant Documents Found (All Simulations)', fontsize=14)
        ax.set_title(f'Merged Recall Curves - {self.dataset_name}\n'
                    f'{self.n_simulations} simulations × {self.n_docs} documents each\n'
                    f'Color indicates dominant mode (Blue=ASReview, Red=Agent)',
                    fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
        
        # Set limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, self.n_relevant * self.n_simulations * 1.05)
        
        # Add text box with basic info
        textstr = f'Total simulations: {self.n_simulations}\n'
        textstr += f'Documents per simulation: {self.n_docs}\n'
        textstr += f'Relevant per simulation: {self.n_relevant}\n'
        textstr += f'Total review events: {len(self.all_events):,}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        
        plt.show()
    
    def plot_individual_curves_colored(self, save_path: str = None):
        """Plot individual simulation curves with color coding by mode"""
        # Use white background style
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Group events by simulation
        sim_events = defaultdict(list)
        for event in self.all_events:
            sim_events[event['sim_id']].append(event)
        
        # Define a color palette for different simulations
        colors = plt.cm.rainbow(np.linspace(0, 1, len(sim_events)))
        
        # Plot each simulation with step function
        for idx, (sim_id, events) in enumerate(sim_events.items()):
            # Sort events by proportion
            events.sort(key=lambda x: x['proportion'])
            
            # Extract x and y values
            x_vals = [0] + [e['proportion'] for e in events]
            y_vals = [0] + [e['n_relevant_found_adjusted'] for e in events]
            
            # Plot as step function with unique color per simulation
            ax.step(x_vals, y_vals, where='post', color=colors[idx], 
                    linewidth=1.5, alpha=0.8)
        
        # Plot random baseline (black step function)
        total_relevant_adj = self.n_relevant - 1
        random_x = np.linspace(0, 1, 100)
        random_y = random_x * total_relevant_adj
        ax.step(random_x, random_y, 'k-', linewidth=2.5, where='post', 
                label='Random', alpha=1.0)
        
        # Formatting
        ax.set_xlabel('Proportion of labeled records', fontsize=16)
        ax.set_ylabel('Recall', fontsize=16)
        
        # Title with specific format
        title = f'{self.dataset_name} -m rf -e doc2vec'
        ax.set_title(title, fontsize=18, pad=20)
        
        # Set axis properties
        ax.set_xlim(0, 1)
        ax.set_ylim(0, self.n_relevant - self.relevant_in_seed if hasattr(self, 'relevant_in_seed') else self.n_relevant)
        
        # Grid
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # Tick parameters
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Make left and bottom spines thicker
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Set background color
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\nPlot saved to: {save_path}")
        
        plt.show()
    
    def plot_synergy_style(self, save_path: str = None):
        """Plot in exact Synergy website style"""
        # Use white background style
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Group events by simulation
        sim_events = defaultdict(list)
        for event in self.all_events:
            sim_events[event['sim_id']].append(event)
        
        # Define a diverse color palette
        # Use tab20 for more distinct colors
        if len(sim_events) <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
        else:
            # For more simulations, combine multiple colormaps
            colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
            colors2 = plt.cm.tab20b(np.linspace(0, 1, 20))
            colors3 = plt.cm.tab20c(np.linspace(0, 1, 20))
            all_colors = np.vstack([colors1, colors2, colors3])
            # Cycle through colors if needed
            colors = all_colors[np.arange(len(sim_events)) % len(all_colors)]
        
        # Plot each simulation with step function
        for idx, (sim_id, events) in enumerate(sim_events.items()):
            # Sort events by proportion
            events.sort(key=lambda x: x['proportion'])
            
            # Extract x and y values
            x_vals = [0] + [e['proportion'] for e in events]
            y_vals = [0] + [e['n_relevant_found_adjusted'] for e in events]
            
            # Plot as step function
            ax.step(x_vals, y_vals, where='post', color=colors[idx % len(colors)], 
                    linewidth=1.2, alpha=0.9)
        
        # Plot random baseline (thick black step function)
        total_relevant_adj = self.n_relevant - 1
        # Create more points for smoother baseline
        n_points = self.n_docs
        random_x = np.linspace(0, 1, n_points)
        random_y = np.round(random_x * total_relevant_adj)
        ax.step(random_x, random_y, 'k-', linewidth=3, where='post', 
                alpha=1.0, zorder=100)  # Put on top
        
        # Formatting exactly like Synergy
        ax.set_xlabel('Proportion of labeled records', fontsize=16, fontfamily='sans-serif')
        ax.set_ylabel('Recall', fontsize=16, fontfamily='sans-serif')
        
        # Title format: Dataset -m model -e embedding
        title = f'{self.dataset_name} -m rf -e tfidf -hybrid(agent model enhanced)'
        ax.set_title(title, fontsize=18, fontfamily='sans-serif', pad=15)
        
        # Set axis properties
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.5, total_relevant_adj + 0.5)
        
        # Grid style
        ax.grid(True, alpha=0.2, linewidth=0.8, color='gray')
        ax.set_axisbelow(True)  # Grid behind plots
        
        # Tick parameters
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Frame style
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')
        
        # Set background color
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            synergy_save_path = save_path.replace('.png', '_synergy_style.png')
            plt.savefig(synergy_save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"\nSynergy-style plot saved to: {synergy_save_path}")
        
        plt.show()


def main():
    """Main function to run batch simulations"""
    
    # Simulation parameters - modify these as needed
    params = {
        'dataset_name': "Appenzeller-Herzog_2019",  # Dataset name
        'n_simulations': 65,                        # Number of simulations to run
        'switch_threshold': 0.15,                    # percentage of consecutive irrelevant triggers switch
        'n_layers': 20,                               # Number of pyramid layers
        'docs_per_agent': 10,                        # Documents per agent
        'n_important_features': 20,                  # Number of important features from RF
        'random_seed': 42                            # Random seed for reproducibility
    }
    
    print("\n" + "=" * 60)
    print("ADAPTIVE ASREVIEW-AGENT HYBRID BATCH SIMULATION")
    print("=" * 60)
    
    # Create simulation instance
    sim = BatchSimulation(**params)
    
    # Load dataset
    sim.load_dataset()
    
    # Run simulations
    sim.run_simulations()
    
    # Plot in Synergy website style
    plot_path = f"recall_curves_{params['dataset_name']}_{params['n_simulations']}sims.png"
    sim.plot_synergy_style(save_path=plot_path)
    
    # Also plot the colored version if needed
    # sim.plot_individual_curves_colored(save_path=plot_path)
    
    # Plot merged/aggregated view (optional)
    # merged_plot_path = f"recall_curves_{params['dataset_name']}_{params['n_simulations']}sims_merged.png"
    # sim.plot_merged_recall_curve(save_path=merged_plot_path)
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"\nDisplaying {params['n_simulations']} plots made from "
          f"{params['n_simulations']} simulations.")
    print(f"Dataset: {params['dataset_name']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
