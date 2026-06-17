function X = generate_scenarios_from_marginals(supports, probs, T, dependenceMode, varargin)
%GENERATE_SCENARIOS_FROM_MARGINALS Sample discrete marginals under dependence.
%
% Supported dependence modes:
%   independent, comonotone, group_comonotone, gaussian, t_copula

    opts = parse_options(varargin{:});
    n = numel(supports);

    switch lower(dependenceMode)
        case 'independent'
            U = rand(T,n);

        case 'comonotone'
            u = rand(T,1);
            U = repmat(u, 1, n);

        case 'group_comonotone'
            U = zeros(T,n);
            labels = mod((1:n)-1, opts.numGroups) + 1;
            for g = 1:opts.numGroups
                idx = labels == g;
                u = rand(T,1);
                U(:,idx) = repmat(u, 1, nnz(idx));
            end

        case 'gaussian'
            R = make_equicorr(n, opts.rho);
            Z = randn(T,n) * chol(R);
            U = 0.5 * erfc(-Z ./ sqrt(2));

        case 't_copula'
            R = make_equicorr(n, opts.rho);
            Z = randn(T,n) * chol(R);
            g = sum(randn(T, opts.df).^2, 2);
            Y = Z ./ sqrt(g / opts.df);
            U = 0.5 * erfc(-Y ./ sqrt(2));

        otherwise
            error('Unsupported dependenceMode: %s.', dependenceMode);
    end

    X = zeros(T,n);
    for i = 1:n
        X(:,i) = inverse_discrete_cdf_public(U(:,i), supports{i}, probs{i});
    end
end

function opts = parse_options(varargin)
    opts = struct();
    opts.rho = 0.6;
    opts.df = 4;
    opts.numGroups = 3;

    if mod(numel(varargin), 2) ~= 0
        error('Options must be name-value pairs.');
    end
    for j = 1:2:numel(varargin)
        opts.(varargin{j}) = varargin{j+1};
    end
end

function R = make_equicorr(n, rho)
    R = (1-rho) * eye(n) + rho * ones(n);
end

function x = inverse_discrete_cdf_public(u, support, prob)
    support = support(:);
    prob = prob(:) / sum(prob);
    cdf = cumsum(prob);
    x = zeros(size(u));

    for t = 1:numel(u)
        idx = find(u(t) <= cdf + 1e-12, 1, 'first');
        x(t) = support(idx);
    end
end
