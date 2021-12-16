function g = grad_est_det(obj, x, t, c_det)
    d = length(x);
    g = zeros(d, 1);
    for i = 1:d
        e = zeros(d, 1);
        e(i) = 1;
        g = g + (obj(x + c_det(t)*e) - obj(x))*e/c_det(t);
    end
end