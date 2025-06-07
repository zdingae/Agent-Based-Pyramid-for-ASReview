import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict

class AdaptiveASReviewAgentHybrid:
    """
    Adaptive hybrid approach:
    1. Start with standard ASReview
    2. Switch to agent-based pyramid when hitting sparse regions (5% consecutive irrelevant)
    3. Can switch back to ASReview if finding relevant documents again
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 labels: pd.Series,
                 switch_threshold: float = 0.05,  # 5% consecutive irrelevant
                 n_layers: int = 5,
                 docs_per_agent: int = 50,
                 initial_seed_size: int = 2):
        
        self.data = data
        self.labels = labels
        self.switch_threshold = switch_threshold
        self.n_layers = n_layers
        self.docs_per_agent = docs_per_agent
        self.initial_seed_size = initial_seed_size
        
        # Prepare texts
        self.texts = self._prepare_texts()
        
        # Global tracking
        self.reviewed = set()
        self.relevant_found = set()
        self.review_history = []  # Track order of reviews
        
        # ASReview components
        self.vectorizer = None
        self.features = None
        self.classifier = None
        
        # Agent components
        self.agents = None
        self.agent_mode = False
        
        # Metrics
        self.metrics = {
            'n_reviewed': [],
            'n_relevant_found': [],
            'mode': [],  # 'asreview' or 'agent'
            'precision': [],
            'efficiency': []  # Rolling efficiency
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
        print("\nInitializing ASReview mode...")
        
        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.features = self.vectorizer.fit_transform(self.texts)
        
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
        
        # Review seeds
        for doc_idx in seeds:
            self._review_document(doc_idx, mode='asreview_init')
        
        print(f"  Initialized with {len(seeds)} seed documents, "
              f"found {len(self.relevant_found)} relevant")
    
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
        
        # Update metrics
        self.metrics['n_reviewed'].append(len(self.reviewed))
        self.metrics['n_relevant_found'].append(len(self.relevant_found))
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
        
        # Train classifier if we have enough diversity
        reviewed_list = list(self.reviewed)
        reviewed_labels = [1 if i in self.relevant_found else 0 for i in reviewed_list]
        
        if len(set(reviewed_labels)) >= 2:
            # Train classifier
            try:
                # Use RandomForest for better performance
                self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                X_train = self.features[reviewed_list]
                self.classifier.fit(X_train, reviewed_labels)
                
                # Predict on unreviewed
                X_unreviewed = self.features[unreviewed]
                pred_proba = self.classifier.predict_proba(X_unreviewed)[:, 1]
                
                # Select top candidates
                n_select = min(batch_size, len(unreviewed))
                top_indices = np.argsort(pred_proba)[-n_select:][::-1]
                selected = [unreviewed[i] for i in top_indices]
                
            except:
                # Fallback to random
                n_select = min(batch_size, len(unreviewed))
                selected = np.random.choice(unreviewed, n_select, replace=False).tolist()
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
    
    def _create_agent_pyramid(self):
        """Create pyramid structure with agents for outlier detection"""
        print("\nCreating agent pyramid for outlier detection...")
        
        # Get unreviewed documents
        unreviewed_indices = [i for i in range(len(self.texts)) if i not in self.reviewed]
        
        if not unreviewed_indices:
            return
        
        # Calculate scores for unreviewed documents
        # Use multiple scoring methods
        scores = self._calculate_outlier_scores(unreviewed_indices)
        
        # Sort by score
        sorted_pairs = sorted(zip(unreviewed_indices, scores), 
                            key=lambda x: x[1], reverse=True)
        sorted_indices = [idx for idx, _ in sorted_pairs]
        
        # Create layers
        layer_size = len(sorted_indices) // self.n_layers
        self.agents = []
        
        print(f"  Creating agents for {len(unreviewed_indices)} unreviewed documents")
        
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
        
        print(f"  Created {len(self.agents)} agents across {self.n_layers} layers")
    
    def _calculate_outlier_scores(self, indices: List[int]) -> np.ndarray:
        """Calculate outlier scores for documents"""
        scores = np.zeros(len(indices))
        
        # Method 1: Distance from reviewed documents
        if len(self.reviewed) > 0 and self.features is not None:
            reviewed_features = self.features[list(self.reviewed)]
            unreviewed_features = self.features[indices]
            
            # Calculate average similarity to reviewed docs
            similarities = cosine_similarity(unreviewed_features, reviewed_features)
            avg_similarity = similarities.mean(axis=1)
            
            # Lower similarity = higher outlier score
            distance_scores = 1 - avg_similarity
            scores += distance_scores * 0.6
        
        # Method 2: Keyword scoring for domain-specific outliers
        # keywords = ['wilson', 'copper', 'hepatic', 'treatment', 'patient', 
                  # 'clinical', 'neurologic', 'penicillamine', 'zinc']
        #keywords = ['wilson', 'copper', 'hepatic', 'treatment', 'patient', 
                  #'penicillamine', 'zinc']
        keywords = ['Wilson','copper','hepatic','treatment','patient',
                       'penicillamine','zinc'] #Manual input for the key words
        
        keyword_scores = np.zeros(len(indices))
        for i, doc_idx in enumerate(indices):
            text = self.texts[doc_idx].lower()
            score = sum(1 for kw in keywords if kw in text)
            keyword_scores[i] = score / np.sqrt(len(text.split()) + 1)
        
        # Normalize keyword scores
        if keyword_scores.max() > 0:
            keyword_scores = keyword_scores / keyword_scores.max()
        
        scores += keyword_scores * 0.4
        
        return scores
    
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
        
        # Review batch from agent
        batch_size = min(10, len(agent_docs))
        batch = agent_docs[:batch_size]
        
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
        
        if not self.agent_mode and self.consecutive_irrelevant >= switch_threshold_docs:
            # Switch to agent mode
            print(f"\n*** SWITCHING TO AGENT MODE ***")
            print(f"  Consecutive irrelevant: {self.consecutive_irrelevant}")
            print(f"  Current progress: {len(self.relevant_found)}/{self.labels.sum()} relevant found")
            
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
        
        elif self.agent_mode and self.consecutive_irrelevant == 0:
            # Found relevant in agent mode, consider switching back
            # Check if we have exhausted current agents
            active_agents = [a for a in self.agents if a['is_active']] if self.agents else []
            
            if not active_agents:
                print(f"\n*** SWITCHING BACK TO ASREVIEW MODE ***")
                print(f"  All agents completed")
                print(f"  Current progress: {len(self.relevant_found)}/{self.labels.sum()} relevant found")
                
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
        print("\n" + "="*60)
        print("ADAPTIVE ASREVIEW + AGENT HYBRID")
        print("="*60)
        
        # Start with ASReview
        self._initialize_asreview()
        
        total_relevant = self.labels.sum()
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
            
            # Print progress every 10 iterations
            if iteration % 10 == 0 or found > 0:
                print(f"\nIteration {iteration} [{mode_str}]: "
                      f"found {found} relevant, "
                      f"total: {len(self.relevant_found)}/{total_relevant}, "
                      f"reviewed: {len(self.reviewed)}")
                
                if self.consecutive_irrelevant > 0:
                    print(f"  Consecutive irrelevant: {self.consecutive_irrelevant}")
        
        # Final summary
        self._print_final_summary()
        
        return self.metrics
    
    def _print_final_summary(self):
        """Print comprehensive final summary"""
        total_relevant = self.labels.sum()
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Total documents reviewed: {len(self.reviewed)}/{len(self.texts)}")
        print(f"Total relevant found: {len(self.relevant_found)}/{total_relevant}")
        print(f"Final precision: {len(self.relevant_found)/len(self.reviewed):.3f}")
        
        # Mode switches
        print(f"\nMode switches: {len(self.mode_switches)}")
        for switch in self.mode_switches:
            print(f"  At iteration {switch['iteration']}: "
                  f"switched to {switch['to_mode'].upper()} "
                  f"({switch['relevant_found']}/{total_relevant} found)")
        
        # Performance by mode
        asreview_reviews = sum(1 for m in self.metrics['mode'] if m.startswith('asreview'))
        agent_reviews = sum(1 for m in self.metrics['mode'] if m == 'agent')
        
        asreview_found = 0
        agent_found = 0
        
        for i, (mode, found) in enumerate(zip(self.metrics['mode'], 
                                             self.metrics['n_relevant_found'])):
            if i == 0:
                continue
            if mode.startswith('asreview'):
                asreview_found += (found - self.metrics['n_relevant_found'][i-1])
            elif mode == 'agent':
                agent_found += (found - self.metrics['n_relevant_found'][i-1])
        
        print("\nPerformance by mode:")
        print(f"  ASReview: {asreview_reviews} reviews, found {asreview_found} relevant")
        if asreview_reviews > 0:
            print(f"    Efficiency: {asreview_found/asreview_reviews:.3f}")
        
        print(f"  Agent mode: {agent_reviews} reviews, found {agent_found} relevant")
        if agent_reviews > 0:
            print(f"    Efficiency: {agent_found/agent_reviews:.3f}")
    
    def plot_results(self):
        """Visualize the adaptive hybrid results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Relevant documents found over time
        ax1.plot(self.metrics['n_reviewed'], self.metrics['n_relevant_found'], 
                'b-', linewidth=2)
        
        # Mark mode switches
        for switch in self.mode_switches:
            color = 'red' if switch['to_mode'] == 'agent' else 'green'
            ax1.axvline(x=switch['iteration'], color=color, linestyle='--', 
                       alpha=0.5, label=f"Switch to {switch['to_mode']}")
        
        # Random baseline
        random_y = np.array(self.metrics['n_reviewed']) / len(self.texts) * self.labels.sum()
        ax1.plot(self.metrics['n_reviewed'], random_y, 'gray', linestyle=':', 
                label='Random baseline')
        
        ax1.set_xlabel('Documents Reviewed')
        ax1.set_ylabel('Relevant Documents Found')
        ax1.set_title('Adaptive Hybrid: Document Discovery')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Rolling efficiency
        ax2.plot(self.metrics['n_reviewed'], self.metrics['efficiency'], 
                'g-', linewidth=1.5)
        
        # Mark mode switches
        for switch in self.mode_switches:
            ax2.axvline(x=switch['iteration'], color='red', linestyle='--', alpha=0.3)
        
        # Add threshold line
        ax2.axhline(y=0.05, color='orange', linestyle='--', 
                   label='Switch threshold (5%)')
        
        ax2.set_xlabel('Documents Reviewed')
        ax2.set_ylabel('Rolling Efficiency (last 50 docs)')
        ax2.set_title('Efficiency Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, max(self.metrics['efficiency']) * 1.1)
        
        # 3. Mode distribution
        mode_counts = defaultdict(int)
        for mode in self.metrics['mode']:
            if mode.startswith('asreview'):
                mode_counts['ASReview'] += 1
            else:
                mode_counts['Agent'] += 1
        
        ax3.pie(mode_counts.values(), labels=mode_counts.keys(), autopct='%1.1f%%',
                colors=['skyblue', 'lightcoral'])
        ax3.set_title('Time Spent in Each Mode')
        
        # 4. Consecutive irrelevant tracking
        consecutive_irrelevant_history = []
        current_consecutive = 0
        
        for i in range(len(self.review_history)):
            doc_idx = self.review_history[i]
            if doc_idx in self.relevant_found:
                current_consecutive = 0
            else:
                current_consecutive += 1
            consecutive_irrelevant_history.append(current_consecutive)
        
        ax4.plot(range(len(consecutive_irrelevant_history)), 
                consecutive_irrelevant_history, 'r-', linewidth=1)
        
        # Mark mode switches
        for switch in self.mode_switches:
            ax4.axvline(x=switch['iteration'], color='blue', linestyle='--', alpha=0.5)
        
        # Threshold line
        threshold_docs = int(self.switch_threshold * len(self.texts))
        ax4.axhline(y=threshold_docs, color='green', linestyle='--', 
                   label=f'Switch threshold ({threshold_docs} docs)')
        
        ax4.set_xlabel('Review Order')
        ax4.set_ylabel('Consecutive Irrelevant Documents')
        ax4.set_title('Trigger for Mode Switching')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('adaptive_hybrid_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def test_adaptive_hybrid():
    """Test the adaptive hybrid approach"""
    
    from synergy_dataset import Dataset
    
    print("Loading dataset...")
    d = Dataset("Appenzeller-Herzog_2019")
    data = d.to_frame()
    labels = data['label_included']
    
    # Filter valid documents
    valid_mask = data['title'].notna() | data['abstract'].notna()
    data = data[valid_mask].reset_index(drop=True)
    labels = labels[valid_mask].reset_index(drop=True)
    
    print(f"Dataset: {len(data)} documents, {labels.sum()} relevant")
    
    # Test different configurations (can be used for determine the best switching rule & pyramid structure)
    configs = [
        {'switch_threshold': 0.05, 'n_layers': 5, 'docs_per_agent': 50},
        {'switch_threshold': 0.03, 'n_layers': 6, 'docs_per_agent': 40},
        {'switch_threshold': 0.07, 'n_layers': 4, 'docs_per_agent': 60},
    ]
    
    results = {}
    
    for i, config in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"CONFIGURATION {i+1}: {config}")
        print('='*80)
        
        hybrid = AdaptiveASReviewAgentHybrid(
            data=data,
            labels=labels,
            **config
        )
        
        metrics = hybrid.run()
        hybrid.plot_results()
        
        results[f"config_{i}"] = {
            'config': config,
            'metrics': metrics,
            'hybrid': hybrid
        }
    
    # Compare configurations
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'green', 'red']
    for i, (key, result) in enumerate(results.items()):
        config = result['config']
        metrics = result['metrics']
        
        label = f"Threshold={config['switch_threshold']:.0%}"
        plt.plot(metrics['n_reviewed'], metrics['n_relevant_found'], 
                colors[i], linewidth=2, label=label, marker='o', markersize=2)
    
    # Add pure ASReview baseline (approximate)
    plt.plot([0, len(labels)], [0, labels.sum()], 'gray', linestyle=':', 
            label='Random baseline')
    
    plt.xlabel('Documents Reviewed')
    plt.ylabel('Relevant Documents Found')
    plt.title('Adaptive Hybrid: Configuration Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, len(labels))
    plt.ylim(0, labels.sum() + 1)
    
    plt.savefig('adaptive_hybrid_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


if __name__ == "__main__":
    results = test_adaptive_hybrid()
