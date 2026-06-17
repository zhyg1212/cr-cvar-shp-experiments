clear; clc;

params = struct();
params.seed = 1;

params.n = 20;
params.kMode = 'half';   % k = floor(n/2)

params.aLow = 1;
params.aHigh = 10;
% Use params.aMode = 'beta_mu' to match the numerical-design document:
% a_i = beta * mu_i + epsilon_i.
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
params.T = 500;

params.generateTestScenarios = true;
params.Ttest = 1000;

params.dependenceMode = 'independent';

data = generate_synthetic_shp_data(params);

disp(data.meta.diagnostics);

fprintf('n = %d\n', data.n);
fprintf('k = %d\n', data.k);
fprintf('Max mean error for worst marginals: %.3e\n', ...
    data.meta.diagnostics.maxMeanErrorWorst);
fprintf('Max MAD error for worst marginals: %.3e\n', ...
    data.meta.diagnostics.maxMADErrorWorst);
fprintf('Max mean error for nominal marginals: %.3e\n', ...
    data.meta.diagnostics.maxMeanErrorNominal);
fprintf('Max MAD error for nominal marginals: %.3e\n', ...
    data.meta.diagnostics.maxMADErrorNominal);
