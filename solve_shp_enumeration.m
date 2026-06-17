function sol = solve_shp_enumeration(a, k, dist)
%SOLVE_SHP_ENUMERATION Exact SHP solver by enumerating first-stage sets.
%
% This is intended for small and medium n, useful for validating generated
% data and comparing nominal, CR-SHP, CVaR-CR-SHP, and mixed distributions.

    a = a(:);
    n = numel(a);
    X = dist.support;
    p = dist.prob(:);

    if n > 24
        warning('Enumeration over 2^n first-stage sets may be slow for n = %d.', n);
    end

    bestValue = -inf;
    bestMask = uint64(0);
    totalMasks = uint64(2)^uint64(n);

    for mask = uint64(0):(totalMasks - 1)
        if bit_count(mask, n) > k
            continue;
        end

        sold = mask_to_logical(mask, n);
        value = sum(a(sold)) + expected_recourse_value(X, p, sold, k);

        if value > bestValue
            bestValue = value;
            bestMask = mask;
        end
    end

    sol = struct();
    sol.S = find(mask_to_logical(bestMask, n));
    sol.mask = bestMask;
    sol.objective = bestValue;
    sol.k = k;
end

function val = expected_recourse_value(X, p, sold, k)
    residual = k - nnz(sold);
    if residual <= 0
        val = 0;
        return;
    end

    availX = X(:, ~sold);
    topVals = zeros(size(X,1), 1);

    for omega = 1:size(X,1)
        row = sort(availX(omega,:), 'descend');
        topVals(omega) = sum(row(1:min(residual, numel(row))));
    end

    val = p(:)' * topVals;
end

function c = bit_count(mask, n)
    c = 0;
    for i = 1:n
        c = c + bitget(mask, i);
    end
end

function sold = mask_to_logical(mask, n)
    sold = false(n,1);
    for i = 1:n
        sold(i) = bitget(mask, i) == 1;
    end
end
