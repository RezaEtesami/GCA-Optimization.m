# GCA-Optimization.m
This project introduces the Gaussian Combined Arms (GCA) algorithm, a metaheuristic optimization method designed for solving complex multi-objective and high-dimensional problems.
% Set parameters
d = 30;  % Dimensionality of the problem
n = 50;  % Initial population size
iter = 500;  % Number of iterations
a = -100;  % Lower bound for truncation
b = 100;   % Upper bound for truncation

% Define the objective function
f = @(x) sum(x.^2);

% Adjust population size excluding air forces (Ground Forces + Air Forces)
n = n - d;  

% Generate initial population within bounds
X = a + (b - a) * rand(n + d, d);

% Initialize personal best positions and their objective values
pbest = X;
pbestobj = arrayfun(@(i) f(pbest(i, :)), 1:size(pbest, 1));

% Identify the global best position and value
[~, gbest_idx] = min(pbestobj);
gbest = pbest(gbest_idx, :);

% Main optimization loop
for iteration = 1:iter
    % Update Ground Forces (GF)
    GFbest = repmat(gbest, n, 1);
    sigma_GF = abs(GFbest - pbest(1:n, :));
    mu_GF = (2 / 3) * GFbest + (1 / 3) * pbest(1:n, :);
    TN_GF = truncnorm(mu_GF, sigma_GF, a, b);

    % Update Air Forces (AF)
    AFbest = repmat(gbest, d, 1);
    sigma_AF = abs(AFbest - pbest(n+1:n+d, :)) / 3;
    mu_AF = (2 / 3) * AFbest + (1 / 3) * pbest(n+1:n+d, :);
    TN_AF = truncnorm(mu_AF, sigma_AF, a, b);

    % Combine forces to create new population
    X = [TN_GF; TN_AF];

    % Evaluate new population
    obj = arrayfun(@(i) f(X(i, :)), 1:size(X, 1));

    % Update personal bests where necessary
    improved_idx = obj <= pbestobj;
    pbest(improved_idx, :) = X(improved_idx, :);
    pbestobj(improved_idx) = obj(improved_idx);

    % Update global best if a better solution is found
    [~, gbest_idx] = min(pbestobj);
    gbest = pbest(gbest_idx, :);
end

% Print the best solution found
disp('Best solution found:');
disp(gbest);

% Print the objective value of the best solution found
best_obj_value = f(gbest);
fprintf('Objective value of the best solution: %.20f\n', best_obj_value);

% Function for truncated normal distribution
function samples = truncnorm(mu, sigma, a, b)
    % Number of samples to generate
    n = numel(mu);
    samples = mu + sigma .* randn(n, size(mu, 2));
    samples(samples < a) = a;
    samples(samples > b) = b;
end
