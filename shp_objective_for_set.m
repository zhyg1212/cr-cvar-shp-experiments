function value = shp_objective_for_set(a, k, dist, sold)
%SHP_OBJECTIVE_FOR_SET Objective value of a fixed first-stage set.

    a = a(:);
    X = dist.support;
    p = dist.prob(:);
    sold = sold(:) > 0;

    residual = k - nnz(sold);
    firstStage = sum(a(sold));

    if residual <= 0
        value = firstStage;
        return;
    end

    recourse = zeros(size(X,1),1);
    for omega = 1:size(X,1)
        vals = sort(X(omega, ~sold), 'descend');
        recourse(omega) = sum(vals(1:min(residual, numel(vals))));
    end

    value = firstStage + p' * recourse;
end
