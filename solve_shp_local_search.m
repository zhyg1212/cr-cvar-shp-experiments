function sol = solve_shp_local_search(a, k, dist, opts)
%SOLVE_SHP_LOCAL_SEARCH Improve Algorithm 2 by add/drop/swap local search.
%
% The method is intended for numerical policy comparison. It starts from
% the approximation solution and a few simple alternatives, then repeatedly
% applies the best improving add/drop/swap move under the supplied training
% distribution.

    if nargin < 4
        opts = struct();
    end
    opts = set_default_local(opts, 'maxIter', 50);
    opts = set_default_local(opts, 'tol', 1e-8);
    opts = set_default_local(opts, 'numRandomStarts', 3);

    a = a(:);
    n = numel(a);

    starts = {};

    approxSol = solve_shp_approximation(a, k, dist);
    starts{end+1} = mask_from_set(approxSol.S, n); %#ok<AGROW>

    [~, ordA] = sort(a, 'descend');
    sFirst = false(n,1);
    sFirst(ordA(1:k)) = true;
    starts{end+1} = sFirst; %#ok<AGROW>

    starts{end+1} = false(n,1); %#ok<AGROW>

    for r = 1:opts.numRandomStarts
        sRand = false(n,1);
        perm = randperm(n);
        card = randi([0,k]);
        sRand(perm(1:card)) = true;
        starts{end+1} = sRand; %#ok<AGROW>
    end

    bestSold = false(n,1);
    bestValue = -inf;

    for s = 1:numel(starts)
        [sold, value] = improve_one_start(a, k, dist, starts{s}, opts);
        if value > bestValue
            bestValue = value;
            bestSold = sold;
        end
    end

    sol = struct();
    sol.S = find(bestSold);
    sol.objective = bestValue;
    sol.k = k;
    sol.method = 'approximation initialized local search';
end

function [sold, currentValue] = improve_one_start(a, k, dist, sold, opts)
    n = numel(a);
    currentValue = shp_objective_for_set(a, k, dist, sold);

    for iter = 1:opts.maxIter %#ok<NASGU>
        bestMoveValue = currentValue;
        bestMoveSold = sold;

        if nnz(sold) < k
            notSold = find(~sold);
            for idx = 1:numel(notSold)
                trial = sold;
                trial(notSold(idx)) = true;
                val = shp_objective_for_set(a, k, dist, trial);
                if val > bestMoveValue + opts.tol
                    bestMoveValue = val;
                    bestMoveSold = trial;
                end
            end
        end

        soldIdx = find(sold);
        for idx = 1:numel(soldIdx)
            trial = sold;
            trial(soldIdx(idx)) = false;
            val = shp_objective_for_set(a, k, dist, trial);
            if val > bestMoveValue + opts.tol
                bestMoveValue = val;
                bestMoveSold = trial;
            end
        end

        soldIdx = find(sold);
        notSold = find(~sold);
        for ii = 1:numel(soldIdx)
            for jj = 1:numel(notSold)
                trial = sold;
                trial(soldIdx(ii)) = false;
                trial(notSold(jj)) = true;
                val = shp_objective_for_set(a, k, dist, trial);
                if val > bestMoveValue + opts.tol
                    bestMoveValue = val;
                    bestMoveSold = trial;
                end
            end
        end

        if bestMoveValue <= currentValue + opts.tol
            break;
        end

        sold = bestMoveSold;
        currentValue = bestMoveValue;
    end
end

function sold = mask_from_set(S, n)
    sold = false(n,1);
    sold(S) = true;
end

function s = set_default_local(s, fieldName, defaultValue)
    if ~isfield(s, fieldName) || isempty(s.(fieldName))
        s.(fieldName) = defaultValue;
    end
end
