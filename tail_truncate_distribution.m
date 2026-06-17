function tailDist = tail_truncate_distribution(dist, alpha)
%TAIL_TRUNCATE_DISTRIBUTION Retain the lower alpha probability mass.
%
% The chain support is assumed to be ordered from high componentwise values
% to low componentwise values, as returned by build_chain_worst_distribution.

    if alpha <= 0 || alpha > 1
        error('alpha must be in (0,1].');
    end

    X = dist.support;
    p = dist.prob(:);
    m = numel(p);

    newP = zeros(m,1);
    remaining = alpha;

    for j = m:-1:1
        take = min(p(j), remaining);
        newP(j) = take / alpha;
        remaining = remaining - take;
        if remaining <= 1e-12
            break;
        end
    end

    keep = newP > 1e-12;
    tailDist = struct();
    tailDist.support = X(keep,:);
    tailDist.prob = newP(keep) / sum(newP(keep));
    tailDist.supportSize = nnz(keep);
    tailDist.alpha = alpha;
end
