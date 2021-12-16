function c1 = c(t, d, m, mode)

    if mode == 1
        c1 = 2/(d^(3/2)*(t + 8)^(1/3));
    elseif mode == 2
        c1 = 2*sqrt(m)/(d^(3/2)*(t + 8)^(1/3));
    elseif mode == 0
        c1 = 2/(d^(1/2)*(t + 8)^(1/3));
    end
end