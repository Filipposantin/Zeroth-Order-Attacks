% FRANK WOLF ALGORIHTM
%
% obj = objective function
% sample = istance we want to attack
% b = number of gradient directions (only for I-RDSA)
% step_size = function that regulates stepsize
% delta = sampling parameter
% beta = momentum parameter
% epsilon = maximal distortion
% T = number of iterations
% mode = 0: uniform distribution for gradient estimation, 1: gaussian
% distribution

function [x, loss_history] = FW_black(obj, sample, b, ...
step_size,delta, beta, epsilon, T, mode)
    m = grad_est(obj, sample, b, delta, mode);
    x = sample;
    loss_history = obj(x);
    for it  = 1:T
        q = grad_est(obj,x, b, delta, mode);
        m = beta * m + (1 - beta) * q;
        [~, ik] = max(abs(m));
        e = zeros(length(m),1);
        e(ik) = 1;
        v = -epsilon * sign(m(ik)) * e + sample;
        d = v - x;
        x = x + step_size(it)*d;
        loss_history = [loss_history obj(x)];
        if obj(x) < 1e-6
            break;
        end
    end     
end

