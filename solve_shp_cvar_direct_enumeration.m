function sol = solve_shp_cvar_direct_enumeration(a, k, dist, alpha)
%SOLVE_SHP_CVAR_DIRECT_ENUMERATION Exact direct CVaR-SHP solver.

    a = a(:);
    n = numel(a);

    bestValue = -inf;
    bestMask = uint64(0);
    totalMasks = uint64(2)^uint64(n);

    for mask = uint64(0):(totalMasks - 1)
        if bit_count_local(mask, n) > k
            continue;
        end

        sold = mask_to_logical_local(mask, n);
        value = shp_cvar_objective_for_set(a, k, dist, sold, alpha);

        if value > bestValue
            bestValue = value;
            bestMask = mask;
        end
    end

    sol = struct();
    sol.S = find(mask_to_logical_local(bestMask, n));
    sol.mask = bestMask;
    sol.objective = bestValue;
    sol.k = k;
end

function c = bit_count_local(mask, n)
    c = 0;
    for i = 1:n
        c = c + bitget(mask, i);
    end
end

function sold = mask_to_logical_local(mask, n)
    sold = false(n,1);
    for i = 1:n
        sold(i) = bitget(mask, i) == 1;
    end
end
