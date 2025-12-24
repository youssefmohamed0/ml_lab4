import numpy as np

class GMM:
    def __init__(self, n_components, covariance_type='full', convergence_threshold=1e-3, max_iter=100, regularized_covariance=1e-6):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.convergence_threshold = convergence_threshold
        self.max_iter = max_iter

        # Regularization to prevent singular matrices
        self.regularized_covariance = regularized_covariance

        # Parameters to be learned
        self.means = None
        self.covariances = None
        self.weights = None
        self.log_likelihood_history = []
        self.converged = False

    def _initialize_parameters(self, x):
        n_samples, n_features = x.shape

        # Initialize Weights (uniform)
        self.weights = np.full(self.n_components, 1 / self.n_components)

        # Initialize Means randomly
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = x[indices]

        # Initialize Covariances based on type
        if self.covariance_type == 'full':
            self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
        elif self.covariance_type == 'tied':
            self.covariances = np.eye(n_features)
        elif self.covariance_type == 'diagonal':
            self.covariances = np.ones((self.n_components, n_features))
        elif self.covariance_type == 'spherical':
            self.covariances = np.ones(self.n_components)

    def _estimate_gaussian_log_probability(self, x):
        n_samples, n_features = x.shape
        log_probabilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            mean = self.means[k]

            # Compute log-probability based on covariance type
            if self.covariance_type == 'full':
                # Add regularization to diagonal
                covariance_matrix = self.covariances[k] + np.eye(n_features) * self.regularized_covariance
                try:
                    # Cholesky decomposition for stable determinant and inverse
                    lower_triangular = np.linalg.cholesky(covariance_matrix)
                    log_determinant = 2 * np.sum(np.log(np.diag(lower_triangular)))
                    precision_matrix = np.linalg.inv(covariance_matrix)

                    difference = x - mean
                    # Mahalanobis distance calculation (x-mu) * inv(cov) * (x-mu)^T
                    temp = np.dot(difference, precision_matrix)
                    mahalanobis_distance = np.sum(temp * difference, axis=1)

                    log_probabilities[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + log_determinant + mahalanobis_distance)
                except np.linalg.LinAlgError:
                    log_probabilities[:, k] = -np.inf # Handle singular matrix

            elif self.covariance_type == 'tied':
                covariance_matrix = self.covariances + np.eye(n_features) * self.regularized_covariance
                try:
                    lower_triangular = np.linalg.cholesky(covariance_matrix)
                    log_determinant = 2 * np.sum(np.log(np.diag(lower_triangular)))
                    precision_matrix = np.linalg.inv(covariance_matrix)

                    difference = x - mean
                    temp = np.dot(difference, precision_matrix)
                    mahalanobis_distance = np.sum(temp * difference, axis=1)

                    log_probabilities[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + log_determinant + mahalanobis_distance)
                except np.linalg.LinAlgError:
                    log_probabilities[:, k] = -np.inf

            elif self.covariance_type == 'diagonal':
                covariance_vector = self.covariances[k] + self.regularized_covariance
                log_determinant = np.sum(np.log(covariance_vector))

                difference = x - mean
                mahalanobis_distance = np.sum((difference ** 2) / covariance_vector, axis=1)

                log_probabilities[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + log_determinant + mahalanobis_distance)

            elif self.covariance_type == 'spherical':
                covariance_scalar = self.covariances[k] + self.regularized_covariance
                log_determinant = n_features * np.log(covariance_scalar)

                difference = x - mean
                mahalanobis_distance = np.sum((difference ** 2), axis=1) / covariance_scalar

                log_probabilities[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + log_determinant + mahalanobis_distance)

        return log_probabilities

    def _e_step(self, x):
        # Calculate log probabilities: log(P(x|theta))
        log_probabilities = self._estimate_gaussian_log_probability(x)

        # Weighted log probabilities: log(w_k) + log(P(x|theta))
        weighted_log_probabilities = log_probabilities + np.log(self.weights + 1e-10)

        # Log-Sum-Exp trick for numerical stability
        max_weighted_log_prob = np.max(weighted_log_probabilities, axis=1, keepdims=True)

        # Use consistent variable name
        log_responsibilities = weighted_log_probabilities - max_weighted_log_prob
        log_responsibilities = log_responsibilities - np.log(np.sum(np.exp(log_responsibilities), axis=1, keepdims=True))

        # Responsibilities
        responsibilities = np.exp(log_responsibilities)
        return responsibilities, weighted_log_probabilities

    def _m_step(self, x, responsibilities):
        n_samples, n_features = x.shape
        weights_sum = np.sum(responsibilities, axis=0) # Nk

        # Update Weights
        self.weights = weights_sum / n_samples

        # Added 1e-10 to the denominator to prevent DivisionByZero errors if a cluster becomes empty (has 0 responsibilities) during training
        safe_weights_sum = weights_sum[:, np.newaxis] + 1e-10

        # Update Means
        self.means = np.dot(responsibilities.T, x) / safe_weights_sum

        # 3. Update Covariances
        if self.covariance_type == 'full':
            for k in range(self.n_components):
                difference = x - self.means[k]
                weighted_difference = responsibilities[:, k][:, np.newaxis] * difference
                self.covariances[k] = np.dot(weighted_difference.T, difference) / (weights_sum[k] + 1e-10)

        elif self.covariance_type == 'tied':
            average_covariance = np.zeros((n_features, n_features))
            for k in range(self.n_components):
                difference = x - self.means[k]
                weighted_difference = responsibilities[:, k][:, np.newaxis] * difference
                average_covariance += np.dot(weighted_difference.T, difference)
            self.covariances = average_covariance / n_samples

        elif self.covariance_type == 'diagonal':
            for k in range(self.n_components):
                difference = x - self.means[k]
                # Only keep diagonal elements (variance)
                self.covariances[k] = np.sum(responsibilities[:, k][:, np.newaxis] * (difference ** 2), axis=0) / (weights_sum[k] + 1e-10)

        elif self.covariance_type == 'spherical':
            for k in range(self.n_components):
                difference = x - self.means[k]
                # Average variance across all features
                variance = np.sum(responsibilities[:, k][:, np.newaxis] * (difference ** 2)) / (weights_sum[k] + 1e-10)
                self.covariances[k] = variance / n_features

    def fit(self, x):
        self._initialize_parameters(x)

        for i in range(self.max_iter):
            prev_log_likelihood = self.log_likelihood_history[-1] if self.log_likelihood_history else -np.inf

            # Expectation Step
            responsibilities, weighted_log_probabilities = self._e_step(x)

            # Maximization Step
            self._m_step(x, responsibilities)

            # Compute Log-Likelihood
            max_weighted_log_prob = np.max(weighted_log_probabilities, axis=1)
            sum_exponential = np.sum(np.exp(weighted_log_probabilities - max_weighted_log_prob[:, np.newaxis]), axis=1)
            log_likelihood = np.sum(max_weighted_log_prob + np.log(sum_exponential))

            self.log_likelihood_history.append(log_likelihood)

            # Convergence Check
            if abs(log_likelihood - prev_log_likelihood) < self.convergence_threshold:
                self.converged = True
                break

        return self

    def predict(self, x):
        responsibilities, _ = self._e_step(x)
        return np.argmax(responsibilities, axis=1)

    def bic(self, x):
        # Bayesian Information Criterion (Lower is better)
        n_samples, n_features = x.shape
        log_likelihood = self.log_likelihood_history[-1]

        # Count parameters based on covariance type
        if self.covariance_type == 'full':
            covariance_parameters = self.n_components * n_features * (n_features + 1) / 2
        elif self.covariance_type == 'tied':
            covariance_parameters = n_features * (n_features + 1) / 2
        elif self.covariance_type == 'diagonal':
            covariance_parameters = self.n_components * n_features
        elif self.covariance_type == 'spherical':
            covariance_parameters = self.n_components

        n_parameters = (self.n_components * n_features) + covariance_parameters + (self.n_components - 1)

        return -2 * log_likelihood + n_parameters * np.log(n_samples)

    def aic(self, x):
        # Akaike Information Criterion (Lower is better)
        n_samples, n_features = x.shape
        log_likelihood = self.log_likelihood_history[-1]

        # Calculate params
        if self.covariance_type == 'full':
            covariance_parameters = self.n_components * n_features * (n_features + 1) / 2
        elif self.covariance_type == 'tied':
            covariance_parameters = n_features * (n_features + 1) / 2
        elif self.covariance_type == 'diagonal':
            covariance_parameters = self.n_components * n_features
        elif self.covariance_type == 'spherical':
            covariance_parameters = self.n_components

        n_parameters = (self.n_components * n_features) + covariance_parameters + (self.n_components - 1)

        return -2 * log_likelihood + 2 * n_parameters