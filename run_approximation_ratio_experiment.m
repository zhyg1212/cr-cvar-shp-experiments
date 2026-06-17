clear; clc;

% Approximation-quality experiment.
% For small instances we compute exact optimal values by enumeration and
% compare Algorithm 2 against the exact benchmark.

outDir = fullfile(pwd, 'results');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

seeds = 1:8;
nList = [10, 12, 14, 16];
gammaList = [0.2, 0.5, 0.8];
alpha = 0.10;

modelNames = {'CR-SHP', 'CVaR-CR-SHP'};

rows = struct([]);
rowId = 0;

for n = nList
    for gamma = gammaList
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
            tailDist = tail_truncate_distribution(crDist, alpha);

            dists = {crDist, tailDist};

            for mi = 1:numel(modelNames)
                dist = dists{mi};

                tExact = tic;
                exactSol = solve_shp_enumeration(data.a, data.k, dist);
                exactTime = toc(tExact);

                tAlg = tic;
                algSol = solve_shp_approximation(data.a, data.k, dist);
                algTime = toc(tAlg);

                ratio = algSol.objective / exactSol.objective;
                gap = (exactSol.objective - algSol.objective) / exactSol.objective;

                rowId = rowId + 1;
                rows(rowId).n = n; %#ok<SAGROW>
                rows(rowId).gamma = gamma;
                rows(rowId).k = data.k;
                rows(rowId).seed = seed;
                rows(rowId).model = modelNames{mi};
                rows(rowId).supportSize = dist.supportSize;
                rows(rowId).exactValue = exactSol.objective;
                rows(rowId).algValue = algSol.objective;
                rows(rowId).ratio = ratio;
                rows(rowId).gap = gap;
                rows(rowId).theoryBound = max(data.k / data.n, 0.5);
                rows(rowId).exactTime = exactTime;
                rows(rowId).algTime = algTime;

                fprintf('n=%d gamma=%.1f seed=%d model=%s ratio=%.4f exact=%.3fs alg=%.3fs\n', ...
                    n, gamma, seed, modelNames{mi}, ratio, exactTime, algTime);
            end
        end
    end
end

T = struct2table(rows);
save(fullfile(outDir, 'approximation_ratio_results.mat'), 'T');
writetable(T, fullfile(outDir, 'approximation_ratio_results.csv'));

summary = groupsummary(T, {'n','gamma','model'}, {'mean','min'}, ...
    {'ratio','gap','theoryBound','exactTime','algTime','supportSize'});
writetable(summary, fullfile(outDir, 'approximation_ratio_summary.csv'));

disp(summary(:, {'n','gamma','model','mean_ratio','min_ratio', ...
    'mean_theoryBound','mean_exactTime','mean_algTime'}));
