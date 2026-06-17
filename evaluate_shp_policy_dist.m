function metrics = evaluate_shp_policy_dist(a, k, S, dist)
%EVALUATE_SHP_POLICY_DIST Evaluate a policy on a discrete distribution.

    a = a(:);
    n = numel(a);
    sold = false(n,1);
    sold(S) = true;

    X = dist.support;
    p = dist.prob(:);
    residual = k - nnz(sold);
    firstStage = sum(a(sold));

    R = zeros(size(X,1),1);
    for omega = 1:size(X,1)
        if residual > 0
            vals = sort(X(omega, ~sold), 'descend');
            recourse = sum(vals(1:min(residual, numel(vals))));
        else
            recourse = 0;
        end
        R(omega) = firstStage + recourse;
    end

    metrics = struct();
    metrics.mean = p' * R;
    metrics.cvar05 = weighted_lower_tail_cvar(R, p, 0.05);
    metrics.cvar10 = weighted_lower_tail_cvar(R, p, 0.10);
    metrics.q05 = weighted_quantile(R, p, 0.05);
    metrics.worst = min(R);
end

function v = weighted_lower_tail_cvar(x, p, alpha)
    [x, order] = sort(x(:), 'ascend');
    p = p(order);
    remaining = alpha;
    total = 0;

    for i = 1:numel(x)
        take = min(p(i), remaining);
        total = total + take * x(i);
        remaining = remaining - take;
        if remaining <= 1e-12
            break;
        end
    end

    v = total / alpha;
end

function q = weighted_quantile(x, p, tau)
    [x, order] = sort(x(:), 'ascend');
    p = p(order);
    c = cumsum(p);
    idx = find(c >= tau, 1, 'first');
    q = x(idx);
end
