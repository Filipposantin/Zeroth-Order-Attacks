new_images_fwbb = [];
attack_rate_fwbb = 0;
time_fwbb = 0;
ind_fwbb = [];
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
    [x, loss_history] = FW_black(obj, atan_sample, 20, ...
        @(t) 0.1/sqrt(t + 1),0.01, 0.99, 60, 1500, 0);
    new_images_fwbb = [new_images_fwbb th(x)];
    [~, O_class] = max(model(sample));
    [~, N_class] = max(model(th(x)));
    if O_class ~= N_class
        attack_rate_fwbb = attack_rate_fwbb + 1;
        time_fwbb = time_fwbb +toc();
        ind_fwbb = [ind_fwbb i];
    end
    c = c + 1;
    if mod(c, 5) == 0
        fprintf('%d\n',c)
        fprintf('%d\n',attack_rate_fwbb/c*100)
    end    
end
attack_rate_fwbb = attack_rate_fwbb/100;
save('attack_rate_fwbb.mat','attack_rate_fwbb');
save('new_images_fwbb.mat','new_images_fwbb');
save('time_fwbb.mat','time_fwbb')
save('ind_fwbb.mat','ind_fwbb')