import numpy as np

class ClusteringMetrics:
    @staticmethod
    def euclidean_distance(p1, p2):
        # Using linalg.norm for faster calculation than manual sqrt(sum)
        return np.linalg.norm(p1 - p2)

    def silhouette_score(self, X, labels):
        """Mean Silhouette Coefficient (as implemented previously)"""
        n_samples = X.shape[0]
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2: return 0

        scores = []
        for i in range(n_samples):
            # a(i): Cohesion
            current_label = labels[i]
            same_cluster = X[labels == current_label]
            if len(same_cluster) > 1:
                # Distances to all others in same cluster excluding self
                a_i = np.mean([self.euclidean_distance(X[i], other) for other in same_cluster if not np.array_equal(X[i], other)])
            else:
                a_i = 0

            # b(i): Separation
            b_i = float('inf')
            for other_label in unique_labels:
                if other_label == current_label: continue
                other_cluster = X[labels == other_label]
                avg_dist = np.mean([self.euclidean_distance(X[i], other) for other in other_cluster])
                b_i = min(b_i, avg_dist)

            scores.append((b_i - a_i) / max(a_i, b_i))
        return np.mean(scores)

    def davies_bouldin_index(self, X, labels):
        """
        Computes the Davies-Bouldin Index.
        Lower values indicate better clustering.
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        if n_clusters < 2: return 0

        # 1. Calculate centroids and cluster dispersions (S_i)
        centroids = []
        dispersions = []

        for label in unique_labels:
            cluster_data = X[labels == label]
            centroid = np.mean(cluster_data, axis=0)
            centroids.append(centroid)

            # Dispersion S_i: Average distance from each point in cluster to centroid
            s_i = np.mean([self.euclidean_distance(p, centroid) for p in cluster_data])
            dispersions.append(s_i)

        # 2. Calculate R_ij = (S_i + S_j) / d(C_i, C_j)
        # and find the maximum R_ij for each i
        d_i = []
        for i in range(n_clusters):
            similarities = []
            for j in range(n_clusters):
                if i != j:
                    # Distance between centroids
                    dist_centroids = self.euclidean_distance(centroids[i], centroids[j])
                    # Similarity ratio
                    r_ij = (dispersions[i] + dispersions[j]) / dist_centroids # higher ratio indicates similarity between the 2
                    similarities.append(r_ij)

            # For each cluster, we only care about its "worst" neighbor (highest ratio)
            d_i.append(np.max(similarities))

        # 3. DBI is the average of these maximum similarities
        # A lower DBI value indicates better clustering (tight clusters that are far apart).
        return np.mean(d_i)
    def calinski_harabasz_index(self, X, labels):
        """
        Higher is better: Ratio of between-cluster dispersion to within-cluster dispersion.
        """
        n_samples, n_features = X.shape
        unique_labels = np.unique(labels)
        k = len(unique_labels)

        if k < 2: return 0

        # Global centroid (mean of all data)
        extra_centroid = np.mean(X, axis=0)

        # Between-group dispersion (BCSS)
        # Within-group dispersion (WCSS)
        bcss = 0
        wcss = 0

        for label in unique_labels:
            cluster_data = X[labels == label]
            cluster_centroid = np.mean(cluster_data, axis=0)
            n_i = cluster_data.shape[0]

            # BCSS: n_i * ||centroid_i - global_centroid||^2
            bcss += n_i * np.sum((cluster_centroid - extra_centroid) ** 2)

            # WCSS: sum(||x - centroid_i||^2)
            wcss += np.sum((cluster_data - cluster_centroid) ** 2)

        # Handle edge case for zero WCSS (perfect clusters)
        if wcss == 0: return 0

        # Final CH Formula: (BCSS / (k-1)) / (WCSS / (n - k))
        score = (bcss / (k - 1)) / (wcss / (n_samples - k))
        return score
    
    def within_cluster_sum(self, final_centroids, X, clusters):
        wcss = 0
        for idx, centroid in enumerate(final_centroids):
            # Filter points belonging to the current cluster
            cluster_points = X[clusters == idx]
            # Sum of squared Euclidean distances
            if len(cluster_points) > 0:
                distances = np.sum((cluster_points - centroid) ** 2)
                wcss += distances
        return wcss
    
    def comb2(self,n):
        """Calculates n choose 2, which is n*(n-1)/2."""
        return n * (n - 1) / 2 if n >= 2 else 0

    def adjusted_rand_index_scratch(self,labels_true, labels_pred):
        labels_true = np.array(labels_true)
        labels_pred = np.array(labels_pred)
        n = len(labels_true)
        
        # 1. Get unique classes and clusters
        classes = np.unique(labels_true)
        clusters = np.unique(labels_pred)
        
        # 2. Build the Contingency Table (n_ij)
        # Rows represent true classes, columns represent predicted clusters
        contingency = np.zeros((len(classes), len(clusters)))
        for i, c in enumerate(classes):
            for j, k in enumerate(clusters):
                contingency[i, j] = np.sum((labels_true == c) & (labels_pred == k))
        
        # 3. Calculate the sums of combinations
        sum_comb_nij = np.sum([self.comb2(cell) for cell in contingency.flatten()])
        sum_comb_ai = np.sum([self.comb2(row_sum) for row_sum in np.sum(contingency, axis=1)])
        sum_comb_bj = np.sum([self.comb2(col_sum) for col_sum in np.sum(contingency, axis=0)])
        
        # 4. Apply the ARI Formula
        expected_index = (sum_comb_ai * sum_comb_bj) / self.comb2(n)
        max_index = 0.5 * (sum_comb_ai + sum_comb_bj)
        
        # Handle the case where denominator is 0 (all points in one cluster)
        if max_index == expected_index:
            return 0.0
            
        ari = (sum_comb_nij - expected_index) / (max_index - expected_index)
        return ari

    def calculate_purity(self, labels_true, labels_pred):
        """
        Calculates the purity score for clustering results.
        
        Parameters:
        labels_true (list-like): Ground truth class labels.
        labels_pred (list-like): Cluster labels to be evaluated.
        
        Returns:
        float: Purity score between 0.0 and 1.0 (1.0 is perfect).
        """
        labels_true = np.array(labels_true)
        labels_pred = np.array(labels_pred)
        
        # 1. Get unique cluster IDs
        clusters = np.unique(labels_pred)
        total_samples = len(labels_true)
        
        total_correct = 0
        
        # 2. For each cluster, find the most frequent ground truth label
        for k in clusters:
            # Get ground truth labels for points in the current cluster
            indices = np.where(labels_pred == k)
            ground_truth_in_cluster = labels_true[indices]
            
            if len(ground_truth_in_cluster) == 0:
                continue
                
            # Find the count of the most common class in this cluster
            # unique_counts[1] gives the frequencies
            _, counts = np.unique(ground_truth_in_cluster, return_counts=True)
            total_correct += np.max(counts)
        
        # 3. Calculate final purity
        purity_score = total_correct / total_samples
        return purity_score