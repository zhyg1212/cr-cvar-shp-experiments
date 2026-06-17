function dist = build_chain_worst_distribution(supports, probs)
%BUILD_CHAIN_WORST_DISTRIBUTION Construct the chain-supported worst coupling.
%
% Input:
%   supports{i}: row/column vector of ordered marginal support values.
%   probs{i}:    probabilities for supports{i}.
%
% Output:
%   dist.support: m-by-n scenario matrix.
%   dist.prob:    m-by-1 scenario probabilities.
%
% The construction sweeps common quantile breakpoints. It is the discrete
% comonotone coupling used by the correlation-robust reformulation.

    n = numel(supports);
    breaks = [0; 1];

    sortedSupports = cell(n,1);
    sortedProbs = cell(n,1);
    sortedCdf = cell(n,1);

    for i = 1:n
        s = supports{i}(:);
        p = probs{i}(:);
        if abs(sum(p) - 1) > 1e-8
            p = p / sum(p);
        end

        [sDesc, order] = sort(s, 'descend');
        pDesc = p(order);
        cdf = cumsum(pDesc);

        sortedSupports{i} = sDesc;
        sortedProbs{i} = pDesc;
        sortedCdf{i} = cdf;
        breaks = [breaks; cdf(:)]; %#ok<AGROW>
    end

    breaks = unique(round(breaks, 12));
    breaks = sort(breaks);

    widths = diff(breaks);
    keep = widths > 1e-12;
    left = breaks(1:end-1);
    widths = widths(keep);
    left = left(keep);

    m = numel(widths);
    X = zeros(m, n);

    for j = 1:m
        t = left(j) + widths(j) / 2;
        for i = 1:n
            idx = find(t <= sortedCdf{i} + 1e-12, 1, 'first');
            X(j,i) = sortedSupports{i}(idx);
        end
    end

    dist = struct();
    dist.support = X;
    dist.prob = widths(:) / sum(widths);
    dist.supportSize = m;
end
