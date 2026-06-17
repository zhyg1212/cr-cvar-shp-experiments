clear; clc;

% Systematic numerical experiment using the approximation algorithm.
% The script prints aggregate tables and saves raw results to .mat/.csv.

outDir = fullfile(pwd, 'results');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

seeds = 1:10;
nList = [100, 200];
gammaList = [0.2, 0.5];
alpha = 0.10;
lambda = 0.50;
Tnom = 1000;
Ttest = 5000;

testModes = {'independent', 'gaussian06', 'comonotone', 't_copula06'};
modelNames = {'Nominal', 'CR-SHP', 'CVaR-CR-SHP', 'Mixed'};

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
            params.beta = 0.75;
            params.epsilonLow = -5;
            params.epsilonHigh = 5;
            params.muLow = 50;
            params.muHigh = 100;
            params.lowerGapLow = 10;
            params.lowerGapHigh = 45;
            params.upperGapLow = 10;
            params.upperGapHigh = 45;
            params.dLowerRatioLow = 0.35;
            params.dLowerRatioHigh = 0.70;
            params.deltaRatioLow = 0.75;
            params.deltaRatioHigh = 0.95;
            params.nominalMADMode = 'middle';
            params.generateNominalScenarios = false;
            params.generateTestScenarios = false;

            data = generate_synthetic_shp_data(params);

            Xnom = generate_scenarios_from_marginals(data.supports, data.pNominal, ...
                Tnom, 'independent');
            nominalDist = make_empirical_dist(Xnom);
            crDist = build_chain_worst_distribution(data.supports, data.pWorst);
            tailDist = tail_truncate_distribution(crDist, alpha);
            mixedDist = mix_distributions(nominalDist, 1-lambda, tailDist, lambda);

            sols = cell(4,1);
            sols{1} = solve_shp_approximation(data.a, data.k, nominalDist);
            sols{2} = solve_shp_approximation(data.a, data.k, crDist);
            sols{3} = solve_shp_approximation(data.a, data.k, tailDist);
            sols{4} = solve_shp_approximation(data.a, data.k, mixedDist);

            for tm = 1:numel(testModes)
                mode = testModes{tm};
                Xtest = build_test_scenarios(data, Ttest, mode);

                meanVals = zeros(4,1);
                metrics = cell(4,1);
                for mi = 1:4
                    metrics{mi} = evaluate_shp_policy(data.a, data.k, sols{mi}.S, Xtest);
                    meanVals(mi) = metrics{mi}.mean;
                end
                oracleMean = max(meanVals);

                for mi = 1:4
                    rowId = rowId + 1;
                    rows(rowId).n = n; %#ok<SAGROW>
                    rows(rowId).gamma = gamma;
                    rows(rowId).k = data.k;
                    rows(rowId).seed = seed;
                    rows(rowId).testMode = mode;
                    rows(rowId).model = modelNames{mi};
                    rows(rowId).trainObj = sols{mi}.objective;
                    rows(rowId).testMean = metrics{mi}.mean;
                    rows(rowId).cvar10 = metrics{mi}.cvar10;
                    rows(rowId).q05 = metrics{mi}.q05;
                    rows(rowId).worst = metrics{mi}.worst;
                    rows(rowId).regretToBest = oracleMean - metrics{mi}.mean;
                    rows(rowId).supportCR = crDist.supportSize;
                    rows(rowId).supportTail = tailDist.supportSize;
                    rows(rowId).selectedCount = numel(sols{mi}.S);
                end
            end
        end
    end
end

T = struct2table(rows);
save(fullfile(outDir, 'batch_approx_results.mat'), 'T');
writetable(T, fullfile(outDir, 'batch_approx_results.csv'));

summary = groupsummary(T, {'n','gamma','testMode','model'}, 'mean', ...
    {'testMean','cvar10','q05','worst','regretToBest','supportCR','supportTail'});
writetable(summary, fullfile(outDir, 'batch_approx_summary.csv'));

disp(summary(:, {'n','gamma','testMode','model','mean_testMean','mean_cvar10','mean_q05','mean_regretToBest'}));

function dist = make_empirical_dist(X)
    dist = struct();
    dist.support = X;
    dist.prob = ones(size(X,1),1) / size(X,1);
    dist.supportSize = size(X,1);
end

function X = build_test_scenarios(data, Ttest, mode)
    switch mode
        case 'independent'
            X = generate_scenarios_from_marginals(data.supports, data.pNominal, ...
                Ttest, 'independent');
        case 'gaussian06'
            X = generate_scenarios_from_marginals(data.supports, data.pNominal, ...
                Ttest, 'gaussian', 'rho', 0.6);
        case 'comonotone'
            X = generate_scenarios_from_marginals(data.supports, data.pNominal, ...
                Ttest, 'comonotone');
        case 't_copula06'
            X = generate_scenarios_from_marginals(data.supports, data.pNominal, ...
                Ttest, 't_copula', 'rho', 0.6, 'df', 4);
        otherwise
            error('Unsupported test mode: %s.', mode);
    end
end
