function value = shp_cvar_objective_for_set(a, k, dist, sold, alpha)
%SHP_CVAR_OBJECTIVE_FOR_SET Direct lower-tail CVaR objective for fixed S.

    a = a(:);
    X = dist.support;
    p = dist.prob(:);
    sold = sold(:) > 0;

    residual = k - nnz(sold);
    firstStage = sum(a(sold));
    recourse = zeros(size(X,1),1);

    for omega = 1:size(X,1)
        if residual > 0
            vals = sort(X(omega, ~sold), 'descend');
            recourse(omega) = sum(vals(1:min(residual, numel(vals))));
        else
            recourse(omega) = 0;
        end
    end

    value = firstStage + weighted_lower_tail_cvar_local(recourse, p, alpha);
end

function v = weighted_lower_tail_cvar_local(x, p, alpha)
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
