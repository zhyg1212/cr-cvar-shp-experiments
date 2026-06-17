clear; clc;

% Small end-to-end validation for the synthetic-data generator.
% It solves nominal SHP, CR-SHP, CVaR-CR-SHP, and a mixed model exactly by
% enumerating first-stage sets. Keep n modest for this demo.

params = struct();
params.seed = 1;
params.n = 12;
params.kMode = 'half';
params.aLow = 1;
params.aHigh = 10;
params.aMode = 'beta_mu';
params.beta = 0.8;
params.epsilonLow = -5;
params.epsilonHigh = 5;
params.muLow = 2;
params.muHigh = 12;
params.lowerGapLow = 1;
params.lowerGapHigh = 5;
params.upperGapLow = 1;
params.upperGapHigh = 5;
params.dLowerRatioLow = 0.20;
params.dLowerRatioHigh = 0.50;
params.deltaRatioLow = 0.60;
params.deltaRatioHigh = 0.95;
params.nominalMADMode = 'middle';
params.generateNominalScenarios = true;
params.T = 300;
params.generateTestScenarios = true;
params.Ttest = 1000;
params.dependenceMode = 'independent';

data = generate_synthetic_shp_data(params);

nominalDist = struct();
nominalDist.support = data.X_nominal;
nominalDist.prob = ones(size(data.X_nominal,1),1) / size(data.X_nominal,1);
nominalDist.supportSize = size(data.X_nominal,1);

crDist = build_chain_worst_distribution(data.supports, data.pWorst);
tailDist = tail_truncate_distribution(crDist, 0.10);
mixedDist = mix_distributions(nominalDist, 0.50, tailDist, 0.50);

solNom = solve_shp_enumeration(data.a, data.k, nominalDist);
solCR = solve_shp_enumeration(data.a, data.k, crDist);
solCVaR = solve_shp_enumeration(data.a, data.k, tailDist);
solMix = solve_shp_enumeration(data.a, data.k, mixedDist);

models = {'Nominal'; 'CR-SHP'; 'CVaR-CR-SHP'; 'Mixed'};
solutions = {solNom; solCR; solCVaR; solMix};

fprintf('n = %d, k = %d\n', data.n, data.k);
fprintf('CR support size = %d, tail support size = %d\n\n', ...
    crDist.supportSize, tailDist.supportSize);

fprintf('%-14s %12s %12s %12s %12s %12s\n', ...
    'Model', 'TrainObj', 'TestMean', 'CVaR0.10', 'Q0.05', 'Worst');

for i = 1:numel(models)
    met = evaluate_shp_policy(data.a, data.k, solutions{i}.S, data.X_test);
    fprintf('%-14s %12.4f %12.4f %12.4f %12.4f %12.4f\n', ...
        models{i}, solutions{i}.objective, met.mean, met.cvar10, met.q05, met.worst);
end
