% GrassmannAveragesPCA on D = 100 dimensions sample of N = 100000 random Gaussian variables:
D = 100;
N = 100000;
tmp = rand(D); 
sigma = tmp * tmp'; 
X = mvnrnd(zeros(D, 1), sigma, N);
grassmann_pca_basis_vectors = GrassmannAveragesPCA(X);

% Trimming K = 5 percent of the distribution:
grassmann_pca_basis_vectors = GrassmannAveragesPCA(X, 5)

% Getting only the first 3 basis vectors:
grassmannpca_config = {};
grassmannpca_config.max_dimensions = 3;
grassmann_pca_basis_vectors = GrassmannAveragesPCA(X, 0, grassmannpca_config);