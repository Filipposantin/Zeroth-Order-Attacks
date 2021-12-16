function ro1 = ro(t, d, m, mode)
    if mode == 1
        ro1 = 4/(d^(1/3)*(t + 8)^(2/3));
    elseif mode == 2
        ro1 = 4/((1 + d/m)^(1/3)*(t + 8)^(2/3));
    elseif mode == 0
        ro1 = 4/((t + 8)^(2/3));
    end
end