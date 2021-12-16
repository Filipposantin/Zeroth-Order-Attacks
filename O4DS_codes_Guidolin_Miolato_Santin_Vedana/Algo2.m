% STOCHASTIC GRADIENT FREE FRANK WOLF
%
% obj = objective function
% sample = istance we want to attack
% m = number of gradient directions (only for I-RDSA)
% T = number of iterations
% epsilon = maximal distortion
% step_size = function that regulates stepsize
% mode = 0: KWSA, 1:RDSA, 2:I-RDSA

function [x, loss_history] = Algo2(obj, sample, m, T, ...
epsilon, step_size, mode)
    x = sample;
    loss_history = obj(x);
    dim = length(x);
    d = zeros(dim, 1);
    for it = 1:T
        g = grad_est_google(obj, x, it, m , mode);
        d = (1 - ro(it, dim, m , mode))* d + ro(it, dim, m , mode)*g;
        [~, ik] = max(abs(d));
        e = zeros(length(d),1);
        e(ik) = 1;
        v = -epsilon * sign(d(ik)) * e + sample;
        x = (1 - step_size(it))*x + step_size(it)*v;
        loss_history = [loss_history, obj(x)];
        if obj(x) < 1e-6
            break;
        end
    end
end

