function p_m = DSSKE(X, y, C, theta)

%% Function Description

% The goal of this function is to extract domain-specific sentiment
% knowledge, i.e., the domain-specific sentiment word distributions from
% both labeled samples and the contextual similarities mined from massive
% unlabeled samples. This function contains two major steps. First, extract
% the initial sentiment word distributions from labeled samples using PMI
% according to Eq.(1). Second, propagate the initial sentiment word
% distributions along the contextual similarities to compute the final
% domain-specific sentiment word distributions.

%% Input

% X:   a N*D matrix, represents the feature vectors of labeled samples in a specific domain, where N is the number of labeled samples and D is the dimension of feature vector.
% y:   a N*1 vector, represents the sentiment labels of these labeled samples, where +1 for positive sample and -1 for negative sample.
% C:   a D*D vector, represents the contextual similarities among features mined from massive unlabeled samples according to Eq.(2).
% theta:   a non-negative real value, represents the parameter in Eq.(1) which controls the relative importance of contextual similarities.


%% Output

% p_m:   a D*1 vector, represents the domain-specific sentiment word distribution learned by the algorithm.

%%

% First step: extract the initial sentiment word distributions from labeled
% samples using PMI according to Eq.(1).

s = log10((sum(X(y==1,:))+1)./(sum(X(y==-1,:))+1)*sum(y==-1)/sum(y==1));

%%
% Second step: propagate the initial sentiment word distributions along the
% contextual similarities to compute the final domain-specific sentiment
% word distributions.  

% construct the Laplacian matrix of C.
L = diag(sum(C)) - C;

% solve the optimization problem in Eq.(3).
p = zeros(length(s),1);
p_2 = p;
p_1 = p;
loss = norm(p-s)^2+theta*p'*L*p;
loss_history = loss;
step = 1;
k = 0;
while k<100
    k = k+1;
    a = k/(k+3);
    w = (1+a)*p_1 - a*p_2;
    gradient = 2*(w-s)+2*theta*L*w;
    p = w - step*gradient;
    loss_p = norm(p-s)^2+theta*p'*L*p;
    loss_w = norm(w-s)^2+theta*w'*L*w;
    loss_lip = loss_w + gradient'*(p-w) + step*norm(p-w)^2/2;
    while loss_p>loss_lip
       step = step/2;
       p = w - step*gradient;
       loss_p = norm(p-s)^2+theta*p'*L*p;
       loss_lip = loss_w + gradient'*(p-w) + gamma*norm(p-w)^2/2;
    end
    loss_history = [loss_history loss_p];
    p_2 = p_1;
    p_1 = p;
    if (k>1 && abs(loss_history(end)-loss_history(end-1))/abs(loss_history(end))<0.001)
        break;
    end
end

p_m = p;

end