clear; clc;

% Numerical verification of the lower-tail truncated reformulation.
% We compare direct lower-tail CVaR optimization under P* with the
% equivalent expectation optimization under the truncated distribution.

outDir = fullfile(pwd, 'results');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

seeds = 1:8;
nList = [8, 10, 12, 14, 16];
alphaList = [0.05, 0.10, 0.20];
gamma = 0.5;

rows = struct([]);
rowId = 0;

for n = nList
    for alpha = alphaList
        for seed = seeds
            params = struct();
            params.seed = seed;
            params.n = n;
            params.k = max(1, floor(gamma * n));
            params.aMode = 'beta_mu';
            params.beta = 0.95;
            params.epsilonLow = -8;
            params.epsilonHigh = 8;
            params.muLow = 50;
            params.muHigh = 100;
            params.lowerGapLow = 20;
            params.lowerGapHigh = 70;
            params.upperGapLow = 20;
            params.upperGapHigh = 70;
            params.dLowerRatioLow = 0.45;
            params.dLowerRatioHigh = 0.85;
            params.deltaRatioLow = 0.85;
            params.deltaRatioHigh = 0.98;
            params.nominalMADMode = 'middle';
            params.generateNominalScenarios = false;
            params.generateTestScenarios = false;

            data = generate_synthetic_shp_data(params);
            crDist = build_chain_worst_distribution(data.supports, data.pWorst);

            tDirect = tic;
            directSol = solve_shp_cvar_direct_enumeration(data.a, data.k, crDist, alpha);
            directTime = toc(tDirect);

            tTruncBuild = tic;
            tailDist = tail_truncate_distribution(crDist, alpha);
            truncBuildTime = toc(tTruncBuild);

            tTrunc = tic;
            truncSol = solve_shp_enumeration(data.a, data.k, tailDist);
            truncTime = toc(tTrunc);

            objDiff = abs(directSol.objective - truncSol.objective);
            relDiff = objDiff / max(1, abs(directSol.objective));
            setOverlap = numel(intersect(directSol.S, truncSol.S)) / ...
                max(1, numel(union(directSol.S, truncSol.S)));

            rowId = rowId + 1;
            rows(rowId).n = n; %#ok<SAGROW>
            rows(rowId).k = data.k;
            rows(rowId).alpha = alpha;
            rows(rowId).seed = seed;
            rows(rowId).chainSupport = crDist.supportSize;
            rows(rowId).tailSupport = tailDist.supportSize;
            rows(rowId).directValue = directSol.objective;
            rows(rowId).truncatedValue = truncSol.objective;
            rows(rowId).objectiveDiff = objDiff;
            rows(rowId).relativeDiff = relDiff;
            rows(rowId).setOverlap = setOverlap;
            rows(rowId).directTime = directTime;
            rows(rowId).truncBuildTime = truncBuildTime;
            rows(rowId).truncatedTime = truncTime;

            fprintf('n=%d alpha=%.2f seed=%d diff=%.3e direct=%.3fs trunc=%.3fs\n', ...
                n, alpha, seed, objDiff, directTime, truncTime);
        end
    end
end

T = struct2table(rows);
save(fullfile(outDir, 'cvar_reformulation_results.mat'), 'T');
writetable(T, fullfile(outDir, 'cvar_reformulation_results.csv'));

summary = groupsummary(T, {'n','alpha'}, {'mean','max'}, ...
    {'objectiveDiff','relativeDiff','setOverlap','directTime','truncatedTime','chainSupport','tailSupport'});
writetable(summary, fullfile(outDir, 'cvar_reformulation_summary.csv'));

disp(summary(:, {'n','alpha','mean_objectiveDiff','max_objectiveDiff', ...
    'mean_relativeDiff','mean_directTime','mean_truncatedTime','mean_tailSupport'}));
