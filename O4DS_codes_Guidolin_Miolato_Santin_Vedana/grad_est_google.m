% Function that estimates gradients for Algo 2
% obj = objective function
% x = input istance
% m = number of gradient directions (only for I-RDSA)
% t = current iteration
% mode = 0: KWSA, 1:RDSA, 2:I-RDSA
function g = grad_est_google(obj, x, t, m, mode)
    d = length(x);
    g = zeros(d, 1);
    if mode == 0
        for i = 1:d
            e = zeros(d, 1);
            e(i) = 1;
            g = g + (obj(x + c(t, d, m, mode)*e) - obj(x))*e/c(t, d, m, mode);
        end
    elseif mode == 1
        z = normrnd(0,1,d,1);
        g = g + (obj(x + c(t, d, m, mode)*z) - obj(x))* z /c(t, d, m, mode);
    elseif mode == 2
        for i = 1:m
            z = normrnd(0,1,d,1);
            g = g + (obj(x + c(t, d, m, mode) * z) - obj(x))/c(t, d, m, mode)*z;
        end
        g = g/m;
    end
end
    