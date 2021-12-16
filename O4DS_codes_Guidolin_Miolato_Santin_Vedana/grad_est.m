
function q = grad_est(obj, x, b, delta, mode)
    d = length(x);
    q = zeros(d, 1);
    if mode == 0
        for it = 1:b
            u = normrnd(0,1,d,1);
            u = u / norm(u);
            f_plus = obj(x + delta*u);
            f_minus = obj(x - delta*u);
            q = q + d*(f_plus - f_minus) * u/(2*delta*b);
        end
    
    else
        for it = 1:b
            u = -1 + 2 * rand(d, 1);
            u = u / norm(u);
            f_plus = obj(x + delta*u);
            f_minus = obj(x - delta*u);
            q = q + d*(f_plus - f_minus) * u/(2*delta*b);  
        end
    end
    
end
