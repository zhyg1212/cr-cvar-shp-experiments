function metrics = evaluate_shp_policy(a, k, S, X)
%EVALUATE_SHP_POLICY Evaluate a first-stage SHP decision on sampled scenarios.

    a = a(:);
    n = numel(a);
    sold = false(n,1);
    sold(S) = true;
    residual = k - nnz(sold);

    R = zeros(size(X,1), 1);
    firstStage = sum(a(sold));

    for t = 1:size(X,1)
        if residual > 0
            vals = sort(X(t, ~sold), 'descend');
            recourse = sum(vals(1:min(residual, numel(vals))));
        else
            recourse = 0;
        end
        R(t) = firstStage + recourse;
    end

    metrics = struct();
    metrics.mean = mean(R);
    metrics.cvar05 = lower_tail_cvar(R, 0.05);
    metrics.cvar10 = lower_tail_cvar(R, 0.10);
    metrics.q05 = quantile_local(R, 0.05);
    metrics.worst = min(R);
    metrics.samples = R;
end

function v = lower_tail_cvar(x, alpha)
    x = sort(x(:), 'ascend');
    m = numel(x);
    h = alpha * m;
    fullCount = floor(h);
    frac = h - fullCount;

    if fullCount == 0
        v = x(1);
    elseif fullCount >= m
        v = mean(x);
    else
        v = (sum(x(1:fullCount)) + frac * x(fullCount + 1)) / h;
    end
end

function q = quantile_local(x, tau)
    x = sort(x(:), 'ascend');
    idx = max(1, ceil(tau * numel(x)));
    q = x(idx);
end
