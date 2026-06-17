function sol = solve_shp_forward_greedy(a, k, dist, opts)
%SOLVE_SHP_FORWARD_GREEDY Fast distribution-aware greedy improvement.
%
% The method repeatedly adds the item with the largest positive objective
% gain under the supplied distribution, then applies a backward deletion
% pass. It is intended for numerical diagnostics and policy comparison.

    if nargin < 4
        opts = struct();
    end
    opts = set_default_local(opts, 'tol', 1e-8);

    a = a(:);
    n = numel(a);
    sold = false(n,1);
    currentValue = shp_objective_for_set(a, k, dist, sold);

    while nnz(sold) < k
        bestGain = 0;
        bestItem = 0;
        notSold = find(~sold);
        for idx = 1:numel(notSold)
            trial = sold;
            trial(notSold(idx)) = true;
            val = shp_objective_for_set(a, k, dist, trial);
            gain = val - currentValue;
            if gain > bestGain + opts.tol
                bestGain = gain;
                bestItem = notSold(idx);
                bestValue = val; %#ok<NASGU>
            end
        end
        if bestItem == 0
            break;
        end
        sold(bestItem) = true;
        currentValue = shp_objective_for_set(a, k, dist, sold);
    end

    improved = true;
    while improved
        improved = false;
        soldIdx = find(sold);
        for idx = 1:numel(soldIdx)
            trial = sold;
            trial(soldIdx(idx)) = false;
            val = shp_objective_for_set(a, k, dist, trial);
            if val > currentValue + opts.tol
                sold = trial;
                currentValue = val;
                improved = true;
                break;
            end
        end
    end

    sol = struct();
    sol.S = find(sold);
    sol.objective = currentValue;
    sol.k = k;
    sol.method = 'distribution-aware forward/backward greedy';
end

function s = set_default_local(s, fieldName, defaultValue)
    if ~isfield(s, fieldName) || isempty(s.(fieldName))
        s.(fieldName) = defaultValue;
    end
end
