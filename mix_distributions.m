function distMix = mix_distributions(distA, weightA, distB, weightB)
%MIX_DISTRIBUTIONS Concatenate two discrete scenario distributions.

    if weightA < 0 || weightB < 0 || abs(weightA + weightB - 1) > 1e-8
        error('Weights must be nonnegative and sum to one.');
    end

    distMix = struct();
    distMix.support = [distA.support; distB.support];
    distMix.prob = [weightA * distA.prob(:); weightB * distB.prob(:)];
    distMix.prob = distMix.prob / sum(distMix.prob);
    distMix.supportSize = numel(distMix.prob);
end
