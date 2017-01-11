function [similarity] = computeSentimentBasedDomainSimilarity(p_m, p_n)

%% Function Description

% The goal of this function is to compute the domain similarity between two
% domains based on their domain-specific sentiment word distributions.


%% Input

% p_m:   a D*1 vector, represents the domain-specific sentiment word distribution of domain m, where D is the dimension of feature vector.
% p_n:   a D*1 vector, represents the domain-specific sentiment word distribution of domain m.


%% Output

% similarity:   a real value, represents the domain similarity between domains m and n based on their domain-specific sentiment word distributions.

%%

p_m = p_m/norm(p_m);
p_n = p_n/norm(p_n);
similarity = sum(p_m.*p_n);

if similarity<0
    similarity = 0;
end

end