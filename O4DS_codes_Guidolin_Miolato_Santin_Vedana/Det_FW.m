%DETERMINISTIC FRANK WOLF ALGORITHM
% obj = objective function
% sample = istance we want to attack
% L = lipschitz costant
% T = number of iterations
% epsilon = maximal distortion
function [x, loss_history] = Det_FW(obj, sample, L, T, ...
epsilon)
    dim = length(sample);
    step_size = @(t) 0.5/sqrt(t + 1);
    c_det = @(t) L * step_size(t);
    x = sample;
    loss_history = obj(x);
    for it = 1:T
        g = grad_est_det(obj, x, it, c_det);
        [~, ik] = max(abs(g));
        e = zeros(length(g),1);
        e(ik) = 1;
        v = -epsilon * sign(g(ik)) * e + sample;
        x = (1 - step_size(it))*x + step_size(it)*v;

        loss_history = [loss_history, obj(x)];
        if obj(x) < 1e-6
            break;
        end
    end
end