function sol = solve_shp_approximation(a, k, dist)
%SOLVE_SHP_APPROXIMATION Unified approximation algorithm for SHP variants.
%
% This implements the three-candidate procedure used in the paper:
% greedy two-stage, pure first-stage, and pure second-stage. The returned
% first-stage set is the candidate with the largest objective value under
% the supplied reformulated distribution.

    a = a(:);
    n = numel(a);
    X = dist.support;
    p = dist.prob(:);

    meanB = (p' * X)';
    v = max(a, meanB);
    [~, order] = sort(v, 'descend');

    soldG = false(n,1);
    for pos = 1:n
        i = order(pos);
        if nnz(soldG) < k && a(i) > meanB(i)
            soldG(i) = true;
        end
    end

    [~, orderA] = sort(a, 'descend');
    soldFirst = false(n,1);
    soldFirst(orderA(1:k)) = true;

    soldSecond = false(n,1);

    candidates = {soldG, soldFirst, soldSecond};
    values = zeros(3,1);
    for c = 1:3
        values(c) = shp_objective_for_set(a, k, dist, candidates{c});
    end

    [bestValue, bestIdx] = max(values);
    bestSold = candidates{bestIdx};

    sol = struct();
    sol.S = find(bestSold);
    sol.objective = bestValue;
    sol.candidateValues = values;
    candidateNames = {'greedy', 'first-stage', 'second-stage'};
    sol.candidateName = candidateNames{bestIdx};
    sol.k = k;
end
