function [w, W] = CMSC(X, y, domain, p, P, S, alpha1, alpha2, beta, lambda1, lambda2, loss_type)

%% Function Description

% The goal of this function is to train domain-specific sentiment
% classifiers for multiple domains in a collaborative way by exploiting the
% common sentiment knowledge shared among them. Given a small number of
% labeled samples in multiple domains, the domain similarities between
% these domains, the general sentiment knowledge extracted from
% general-purpose sentiment lexicons, and the domain-specific sentiment
% knowledge of each domain extracted from both labeled and unlabeled
% samples, this function will learn a global sentiment model shared by all
% domains to capture the general sentiment knowlege and a domain-specific
% sentiment model for each domain to capture the domain-specific sentiment
% expressions. The final domain-specific sentiment classifier of each
% domain is the combination of the global sentiment model and their
% domain-specific sentiment model.    
%  



%% Input

% X:   a N*D matrix, represents the feature vectors of labeled samples from multiple domains, where N is the number of all labeled samples and D is the dimension of the feature vector.
% y:   a N*1 vector, represents the sentiment labels of these labeled samples, where +1 for positive samples and -1 for negative samples.
% domain:   a N*1 vector, represents the domain index of each labeled sample.
% p:	a D*1 vector, represents the prior sentiment knowledge extracted from general-purpose sentiment lexicons, where +1 for positive sentiment experessions, -1 for negative sentiment experessions, and 0 for others.
% P:	a D*M vector, represents the domain-specific sentiment knowledge of multiple domains, where M is the number of domains to be analyzed. P(:,m) is the domain-specific sentiment knowledge of the m-th domain.
% S:    a M*M vector, represents the domain similarities. S(m,n) represents the domain similarity between domain m and domain n.
% alpha1:  a non-negative real value, controls the relative importance of the prior general sentiment knowledge extracted from general-purpose sentiment lexicons.
% alpha2:  a non-negative real value, controls the relative importance of the domain-specific sentiment knowledge extracted from both labeled and unlabeled samples.
% beta: a non-negative real value, controls the relative importance of domain similarity knowledge.
% lambda1:  a non-negative real value, controls the model complexity.
% lambda2:  a non-negative real value, controls the model sparsity.
% loss_type:  a string, represents the type of loss function used in our approach.


%% Output

% w:   a D*1 vector, represents the global sentiment model shared by multiple domains. 
% W:   a N*D matrix, represents the domain-specific sentiment models of multiple domains, where W(:,m) is the domain-specific sentiment model of domain m. 

%%

D = size(X,2); % D: dimension of feature vector.
M = max(domain); % M: number of domains to be analyzed.

% initialize sentiment classification models.
w = zeros(D,1); % w: the global sentiment model.
W = zeros(D,M); % W: the domain-specific sentiment models of multiple domains. 
w_2 = w;
w_1 = w;
W_2 = W;
W_1 = W;

% compute the initial objective function value.
loss = computeLoss(X, y, domain, p, P, S, alpha1, alpha2, beta, lambda1, w, W, loss_type)+lambda2*(sum(abs(w))+sum(sum(abs(W))));
loss_history = loss;

k = 0;
L = 1;
eta = 2;
MaxIterNum = 1000;
Threshold = 0.001;

while  k<MaxIterNum
    k = k+1;
    a = k/(k+3);
    
    % update the search points.
    v = (1+a)*w_1 - a*w_2;
    V = (1+a)*W_1 - a*W_2;
    
    % compute the partial derivatives of the objective function value with
    % respect to w and W. 
    [gradient_w, gradient_W] = computeGradient(X, y, domain, p, P, S, alpha1, alpha2, beta, lambda1, v, V, loss_type);
    
    % update the approximate points.
    w = v - gradient_w/L;
    W = V - gradient_W/L;
    w(abs(w)<=lambda2/L) = 0;
    W(abs(W)<=lambda2/L) = 0;
    w = sign(w).*(abs(w)-lambda2/L);
    W = sign(W).*(abs(W)-lambda2/L);
    
    g_w_W = computeLoss(X, y, domain, p, P, S, alpha1, alpha2, beta, lambda1, w, W, loss_type);
    g_v_V = computeLoss(X, y, domain, p, P, S, alpha1, alpha2, beta, lambda1, v, V, loss_type);
    g_lip = g_v_V + gradient_w'*(w-v) + L*norm(w-v)^2/2;
    for m = 1:M
        loss_lip = loss_lip + gradient_W(:,m)'*(W(:,m)-V(:,m)) + L*norm(W(:,m)-V(:,m))^2/2;
    end
    
    % update the step size, i.e., 1/L, if it does not satisfy the condition
    % in Eq.(11).
    while g_w_W>g_lip && L<10^(40)
        
        L = L*eta;
        w = v - gradient_w/L;
        W = V - gradient_W/L;
        w(abs(w)<=lambda2/L)=0;
        W(abs(W)<=lambda2/L)=0;
        w = sign(w).*(abs(w)-lambda2/L);
        W = sign(W).*(abs(W)-lambda2/L);
        g_w_W = computeLoss(X, y, domain, p, P, S, alpha1, alpha2, beta, lambda1, w, W, loss_type);
        g_lip = g_v_V + gradient_w'*(w-v) + L*norm(w-v)^2/2;
        for m = 1:M
            g_lip = g_lip + gradient_W(:,m)'*(W(:,m)-V(:,m))+L*norm(W(:,m)-V(:,m))^2/2;
        end
    end
    loss_history = [loss_history g_w_W+lambda2*(sum(abs(w))+sum(sum(abs(W))))];
    w_2 = w_1;
    w_1 = w;
    W_2 = W_1;
    W_1 = W;
    
    % stop updating if stopping criterion is satisfied.
    if (k>1 && abs(loss_history(end)-loss_history(end-1))/abs(loss_history(end))<Threshold)
        break;
    end
end

end


function loss = computeLoss(X, y, domain, p, P, S, alpha1, alpha2, beta, lambda1, w, W, loss_type)

% compute the objective function value of the model of our approach
% (Eq.(7)). 

M = max(domain);
loss = 0;

% compute the objective function value brought by labeled samples.
if strcmp(loss_type,'squared_loss')
    for m = 1:M
        loss = loss + norm(X(domain==m,:)*(w+W(:,m))-y(domain==m))^2;
    end
elseif strcmp(loss_type,'log_loss')
    for m = 1:M
        loss = loss + sum(log(1+exp(-y(domain==m).*(X(domain==m,:)*(w+W(:,m))))));
    end
end

% compute the objective function value brought by the general sentiment
% knowledge. 
loss = loss - alpha1*p'*w;

% compute the objective function value brought by the domain-specific
% sentiment knowledge and domain similarities. 
for m = 1:M
    loss = loss - alpha2*P(:,m)'*(w+W(:,m));
    for n=1:M
        loss = loss + beta*S(m,n)*norm(W(:,m)-W(:,n))^2;
    end
end

% compute the objective function value brought by the L2-norm
% regularization. 
loss = loss + lambda1*(norm(w)^2+sum(sum(W.^2)));

end

function [gradient_w, gradient_W] = computeGradient(X, y, domain, p, P, S, alpha1, alpha2, beta, lambda1, w, W, loss_type)

% compute the partial derivatives of the objective function value with
% respect to w and W.

M = max(domain);
gradient_w = zeros(size(w));
gradient_W = zeros(size(W));

if strcmp(loss_type,'squared_loss')
    for m = 1:M
        gradient = 2*X(domain==m,:)'*((X(domain==m,:)*(w+W(:,m))-y(domain==m)));
        gradient_w = gradient_w + gradient;
        gradient_W(:,m) = gradient_W(:,m) + gradient;
    end
    
elseif strcmp(loss_type,'log_loss')
    for m = 1:M
        gradient =(y(domain==m)./(1+exp(y(domain==m).*(X(domain==m,:)*(w+W(:,m))))))'*X(domain==m,:);
        gradient_w = gradient_w - gradient';
        gradient_W(:,m) = gradient_W(:,m) - gradient';
    end
end

for m = 1:M
    gradient_W(:,m) = gradient_W(:,m) + 4*beta*(sum(S(m,:))*W(:,m)-W*S(m,:)');
end

gradient_w = gradient_w - alpha1*p - alpha2*sum(P,2)+2*lambda1*w;
gradient_W = gradient_W - alpha2*P + 2*lambda1*W;

end
