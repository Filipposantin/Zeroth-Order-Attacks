new_images_ord = [];
attack_rate_ord = 0;
time_ord = 0;
ind_ord = [];

epsilon = 60;  %%!!!
opts.eps_opt = 1e-8;
% opts.n_initial_atoms = 100; %%!!!
opts.verbosity = false;
opts.f_stop = -0.1;
c = 0;
for i = index
    tic()
    sample = images(:,i);
    out2 = @(x) elu(w2*x+b2);
    out3 = @(x) elu(w3*out2(x)+b3);
    model = @(x) elu(w4*out3(x)+b4);
    [m_o, O_class] = max(model(sample));
    f = @(x) log(abs(model(x)));
    g = @(x) x(O_class);
    h = @(x) x([1:(O_class-1), (O_class+1):end]);
    th = @(x) (1+tanh(x)/0.9999999)/(2);
    obj_ord = @(x) max(g(f(th(x)) - max(h(f(th(x))))),-0.1);
    
    d = length(sample);
    atoms = zeros(d, 2*d);
    atan_sample = atanh(((2)*sample-1)*0.9999999);
    for k = 1:d
        atoms(k,k) = epsilon;
        atoms(k,k + d) = -epsilon; 
        atoms(:,k) = atoms(:,k) + atan_sample; %%!!!
        atoms(:,k+d) = atoms(:,k+d) + atan_sample;  %%!!!
    end
    atoms = real(atoms);
    
    i0 = randi(2*d);
    
    [x_ord,y_ord,f_ord,n_f_ord,it_ord,t_elap_ord,flag_ord] = ORD(obj_ord, atoms, i0, opts);
    new_images_ord = [new_images_ord th(x_ord)];
    [~, O_class] = max(model(sample));
    [~, N_class] = max(model(th(x_ord)));
    if O_class ~= N_class
        attack_rate_ord = attack_rate_ord + 1;
        time_ord = time_ord +toc();
        ind_ord = [ind_ord i];
    end
    c = c + 1;
    if mod(c, 5) == 0
        fprintf('%d\n',c)
        fprintf('%d\n',attack_rate_ord/c*100)
    end
end
attack_rate_ord = attack_rate_ord/100;
save('attack_rate_ord.mat','attack_rate_ord');
save('new_images_ord.mat','new_images_ord');
save('time_ord.mat','time_ord')
save('ind_ord.mat','ind_ord')