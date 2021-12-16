new_images_det = [];
attack_rate_det = 0;
time_fw_det = 0;
ind_det = [];
time_tot = 0;
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
    [x, loss_history] = Det_FW(obj, atan_sample, 0.1, 150, 60);    
    new_images_det = [new_images_det th(x)];
    [~, O_class] = max(model(sample));
    [~, N_class] = max(model(th(x)));
    if O_class ~= N_class
        attack_rate_det = attack_rate_det + 1;
        time_fw_det = time_fw_det +toc();
        ind_det = [ind_det i];
    end
    time_tot = time_tot + toc();
    c = c + 1;
    if mod(c, 5) == 0
        fprintf('%d\n',c)
        fprintf('%d\n',attack_rate_det/c*100)
    end
end
attack_rate_det = attack_rate_det/100;
save('attack_rate_det.mat','attack_rate_det');
save('new_images_det.mat','new_images_det');
save('time_fw_det.mat','time_fw_det')
save('ind_det.mat','ind_det')