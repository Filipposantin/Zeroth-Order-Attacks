new_images_zo2 = [];
attack_rate_mode_2 = 0;
time_zo2 = 0;
ind_zo2 = [];
c = 0;
for i = index
    tic()
    sample = images(:,i);
    atan_sample = atanh(((2)*sample-1)*0.9999999);
    out2 = @(x) elu(w2*x+b2);
    out3 = @(x) elu(w3*out2(x)+b3);
    model = @(x) elu(w4*out3(x)+b4);
    th = @(x) (1+tanh(x)/0.9999999)/(2);
    [m_o, O_class] = max(model(sample));
    f = @(x) log(abs(model(x)));
    g = @(x) x(O_class);
    h = @(x) x([1:(O_class-1), (O_class+1):end]);
    obj = @(x) max(g(f(th(x)) - max(h(f(th(x))))),0);
    [x, loss_history] = Algo2(obj, atan_sample, 20, 2000, ...
    60, @(t) 2/(t + 8), 2);
    new_images_zo2 = [new_images_zo2 th(x)];
    [~, O_class] = max(model(sample));
    [~, N_class] = max(model(th(x)));
    if O_class ~= N_class
        attack_rate_mode_2 = attack_rate_mode_2 + 1;
        time_zo2 = time_zo2 +toc();
        ind_zo2 = [ind_zo2 i];
    end
    c = c + 1;
    if mod(c, 5) == 0
        fprintf('%d\n',c)
        fprintf('%d\n',attack_rate_mode_2/c*100)
    end
end
attack_rate_mode_2 = attack_rate_mode_2/100;
save('attack_rate_mode_2.mat','attack_rate_mode_2');
save('new_images_zo2.mat','new_images_zo2');
save('time_zo2.mat','time_zo2')
save('ind_zo2.mat','ind_zo2')