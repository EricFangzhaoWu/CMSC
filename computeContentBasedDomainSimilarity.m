function [similarity] = computeContentBasedDomainSimilarity(X_m, X_n)

%% Function Description

% The goal of this function is to compute the domain similarity between two
% domains based on their textual term distributions according to
% Jensen-Shannon divergence. 


%% Input

% X_m:   a N_m*D matrix, represents the feature vectors of both labeled and unlabeled samples in domain m, where N_m is the number of samples in domain m and D is the dimension of feature vector.
% X_n:   a N_n*D matrix, represents the feature vectors of both labeled and unlabeled samples in domain n, where N_n is the number of samples in domain n.


%% Output

% similarity:   a real value, represents the domain similarity between domains m and n based on their textual term distributions.

%%

d_m = sum(X_m)+1;
d_n = sum(X_n)+1;
d_m = d_m/sum(d_m);
d_n = d_n/sum(d_n);
d_average = (d_m+d_n)/2;
KL_m_n = sum(d_m.*(log2(d_m./d_average)));
KL_n_m = sum(d_n.*(log2(d_n./d_average)));

similarity = 1-(KL_m_n+KL_n_m)/2;

end