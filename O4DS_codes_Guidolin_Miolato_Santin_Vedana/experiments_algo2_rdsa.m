new_images_zo2_1 = [];
attack_rate_mode1 = 0;
time_zo2_1 = 0;
ind_zo2_1 = [];
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
    [x, loss_history] = Algo2(obj, atan_sample, 20, 20000, ...
    60, @(t) 2/(t + 8), 1);
    new_images_zo2_1 = [new_images_zo2_1 th(x)];
    [~, O_class] = max(model(sample));
    [~, N_class] = max(model(th(x)));
    if O_class ~= N_class
        attack_rate_mode1 = attack_rate_mode1 + 1;
        time_zo2_1 = time_zo2_1 +toc();
        ind_zo2_1 = [ind_zo2_1 i];
    end
    c = c + 1;
    if mod(c, 5) == 0
        fprintf('%d\n',c)
        fprintf('%d\n',attack_rate_mode1/c*100)
    end
end
attack_rate_mode1 = attack_rate_mode1/100;
save('attack_rate_mode1.mat','attack_rate_mode1');
save('new_images_zo2_1.mat','new_images_zo2_1');
save('time_zo2_1.mat','time_zo2_1')
save('ind_zo2_1.mat','ind_zo2_1')