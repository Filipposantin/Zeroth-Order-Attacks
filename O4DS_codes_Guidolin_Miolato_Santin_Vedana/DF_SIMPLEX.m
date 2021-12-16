% -------------------------------------------------------------------------
%
% This file is part of ORD, which is a derivative-free solver for
% optimization problems of the following form:
%
%                                 min f(x)
%                            s.t. x in conv{a_1,...,a_m}
%
% where f(x) is a (black-box) continuously differentiable function and
% conv{a_1,...,a_m} is the convex hull of some given vectors a_1,...,a_m,
% called atoms.
%
% -------------------------------------------------------------------------
%
% Reference paper:
%
% A. Cristofari, F. Rinaldi (2021). A Derivative-Free Method for Structured
% Optimization Problems. SIAM Journal on Optimization, 31(2), 1079-1107.
%
% -------------------------------------------------------------------------
%
% Authors:
% Andrea Cristofari (e-mail: andrea.cristofari@unipd.it)
% Francesco Rinaldi (e-mail: rinaldi@math.unipd.it)
%
% Last update of this file:
% July 7th, 2021
%
% Licensing:
% This file is part of ORD.
% ORD is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% ORD is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% You should have received a copy of the GNU General Public License
% along with ORD. If not, see <http://www.gnu.org/licenses/>.
%
% Copyright 2021 Andrea Cristofari, Francesco Rinaldi.
%
% -------------------------------------------------------------------------

function [x,y,f,n_f,it,t_elap,flag,sampling] = DF_SIMPLEX(obj,A,y,opts)

    % This file implements the DF-SIMPLEX algorithm, used at each iteration
    % of ORD to solve the reduced problems. DF-SIMPLEX can also be called
    % as standalone.
    %
    % If DF-SIMPLEX is called as standalone, the last output argument (i.e.,
    % sampling) can be ignored. It is a structure needed to return the polling
    % samples computed in the neighborhood of the final solution. Assume that,
    % before returning the final solution y, the algorithm computed
    % f(A*(y+alpha_1*d_1)), ..., f(A*(y+alpha_m*d_r)) for some stepsizes
    % alpha_1, ..., alpha_r and some directions d_1, ..., d_r; then
    %   sampling.b is the vector [f(A*(y+alpha_1*d_1))-f(A*y)  ...  f(A*(y+alpha_r*d_r))-f(A*y)]',
    %   sampling.v_d is the vector [i_1  ...  i_r]' such that d_h = sign(i_h)*(e_{i_h}-e_j), h = 1,...,r,
    %   sampling.j is the index j used to compute the polling directions d_1, ..., d_r,
    %   sampling.alpha is the vector [alpha_1  ...  alpha_r]',
    
    t0 = tic;
    
    if (nargin < 3)
        error('at least three input arguments are required');
    end
    if (nargin > 4)
        error('at most four input arguments are required');
    end
    
    if (~isa(obj,'function_handle'))
        error('the first input must be a function handle');
    end
    if (~isnumeric(A) || ~isreal(A) || ~ismatrix(A))
        error('the second input must be a real matrix');
    end
    if (~isnumeric(y) || ~isreal(y) || ~iscolumn(y))
        error('the third input must be a real column vector');
    end
    
    % set options
    f = [];
    eps_opt = 1e-4;
    max_n_f = 100*(size(A,1)+1);
    max_it = Inf;
    f_stop = -Inf;
    max_time = Inf;
    alpha_ini = max(5e-1,eps_opt);
    is_alpha_ini_given = false;
    verbosity = true;
    if (nargin == 4)
        if (~isstruct(opts) || ~isscalar(opts))
            error('the fifth input (which is optional) must be a structure');
        end
        opts_field = fieldnames(opts);
        for i = 1:length(opts_field)
            switch char(opts_field(i))
                case 'eps_opt'
                    eps_opt = opts.eps_opt;
                    if (~isnumeric(eps_opt) || ~isreal(eps_opt) || ~isscalar(eps_opt) || eps_opt<0e0)
                       error('''eps_opt'' must be a non-negative real number');
                    end
                case 'max_n_f'
                    max_n_f = floor(opts.max_n_f);
                    if (~isnumeric(max_n_f) || ~isreal(max_n_f) || ~isscalar(max_n_f) || max_n_f<0e0)
                       error('''max_n_f'' must be a non-negative real number');
                    end
                case 'max_it'
                    max_it = floor(opts.max_it);
                    if (~isnumeric(max_it) || ~isreal(max_it) || ~isscalar(max_it) || max_it<0e0)
                       error('''max_it'' must be a non-negative real number');
                    end
                case 'f_stop'
                    f_stop = opts.f_stop;
                    if (~isnumeric(f_stop) || ~isreal(f_stop) || ~isscalar(f_stop))
                       error('''f_stop'' must be a real number');
                    end
                case 'max_time'
                    max_time = opts.max_time;
                    if (~isnumeric(max_time) || ~isreal(max_time) || ~isscalar(max_time) || max_time<=0e0)
                       error('''max_time'' must be a non-negative real number');
                    end
                case 'f0'
                    f = opts.f0;
                    if (~isnumeric(f) || ~isreal(f) || ~isscalar(f))
                       error('''f0'' must be a real number');
                    end
                case 'alpha_ini'
                    alpha_ini = opts.alpha_ini;
                    if (~isnumeric(alpha_ini) || ~isreal(alpha_ini) || ~isscalar(alpha_ini) || alpha_ini<0e0)
                       error('''alpha_ini'' must be a non-negative real number');
                    end
                    is_alpha_ini_given = true;
                case 'verbosity'
                    verbosity = opts.verbosity;
                    if (~islogical(verbosity) || ~isscalar(verbosity))
                       error('''verbosity'' must be a logical');
                    end
                otherwise
                    error(['in the fifth input (which is optional) ''' char(opts_field(i)) ''' is not a valid field name']);
            end
        end
    end
    
    shuffle = true; % to shuffle or not the variables at the beginning of each outer iteration
    
    n = size(A,2);
    
    b_sampling = zeros(2*n,1);
    alpha_sampling = zeros(2*n,1);
    ind_sampling = false(2*n,1);
    
    if (shuffle)
       ind = randperm(n);
    else
       ind = 1:n; 
    end
    ind_i = 0;
    
    tau = 9e-1;
    [~,j] = max(y);
    y = y/norm(y,1);        
    x = A*y;
    if (isempty(f))
        f = obj(x);
        n_f = 1;
    else
        n_f = 0;
    end
    
    if (verbosity)
        fprintf('DF-SIMPLEX starts\n')
        fprintf('%s%i%s%.4e\n','it = ',0,', f = ',f);
    end
    
    % search directions will be normalized
    
    % line search parameters
    if (~is_alpha_ini_given)
        alpha_max = alpha_ini*max(vecnorm(A*(eye(n)-double(1:n==j)')));
    else
        alpha_max = alpha_ini;
    end
    alpha_vec = max(alpha_max,eps_opt)*ones(n,1); % vector of initial stepsizes
    gamma = 1e-6;
    theta = 5e-1; % stepsize reduction factor
    delta = 5e-1; % reciprocal of the stepsize expansion factor
    
    compute_j = false;
    allow_stop = (alpha_max<=eps_opt);
    skip_d = false(2*n,1);
    
    if (f <= f_stop)
        it = 0;
        sampling.b = b_sampling(ind_sampling);
        sampling.v_d = [];
        sampling.alpha = [];
        sampling.j = j;
        flag = 3;
        if (verbosity)
            fprintf('%s\n','target objective value obtained');
        end
        return;
    end
    if (max_n_f <= 0e0)
        it = 0;
        sampling.b = b_sampling(ind_sampling);
        sampling.v_d = [];
        sampling.alpha = [];
        sampling.j = j;
        flag = 1;
        if (verbosity)
            fprintf('%s\n','maximum number of function evaluations reached');
        end
        return;
    end
    if (max_it <= 0e0)
        it = 0;
        sampling.b = b_sampling(ind_sampling);
        sampling.v_d = [];
        sampling.alpha = [];
        sampling.j = j;
        flag = 2;
        if (verbosity)
            fprintf('%s\n','maximum number of iterations reached');
        end
        return;
    end
    t_elap = toc(t0);
    if (t_elap >= max_time)
        it = 0;
        sampling.b = b_sampling(ind_sampling);
        sampling.v_d = [];
        sampling.alpha = [];
        sampling.j = j;
        flag = 4;
        if (verbosity)
            fprintf('%s\n','maximum cpu time exceeded');
        end
        return;
    end
    
    it = 1;
        
    while (true)
        
        if (n_f >= max_n_f)
            flag = 1;
            break;
        end
        
        % select index i
        if (ind_i < n)
            ind_i = ind_i + 1;
            i = ind(ind_i);
            if (i == j)
                if (ind_i < n)
                    ind_i = ind_i + 1;
                    i = ind(ind_i);
                else
                    compute_j = true;
                end
            end
        else
            compute_j = true;
        end
        
        if (compute_j)
                       
            alpha_vec(j) = min(alpha_vec);
            
            % check stopping condition
            alpha_max = max(alpha_vec);
            if (allow_stop && alpha_max<=eps_opt)
                flag = 0;
                break;
            end
            
            if (it >= max_it)
                flag = 2;
                break;
            end
            
            t_elap = toc(t0);
            if (t_elap >= max_time)
                flag = 4;
                break;
            end
            
            if (verbosity)
                fprintf('%s%i%s%.4e%s%i%s%.4e\n','it = ',it,', f = ',f,', n_f = ',n_f,', alpha_max = ',alpha_max);
            end
            
            it = it + 1;
                        
            % shuffle variables
            if (shuffle)
                ind = randperm(n);
            end
            
            % select a new index j
            [~, j_max] = max(y);
            if (y(j) < tau*y(j_max))
                j = j_max;
            end
            compute_j = false;
            
            % select index i
            ind_i = 1;
            i = ind(ind_i);
            if (i == j)
                ind_i = ind_i + 1;
                i = ind(ind_i);
            end
            
            allow_stop = (alpha_max<=eps_opt);
            
        end
        
        if (y(i)>0e0 || y(j)>0e0) % so that at least one direction between
                                  % (e_i-e_j) and (e_j-e_i) is feasible
            
            linesearch_i = true;
            expansion_i = false;
            
            if (y(i) == 0e0)
                if (~skip_d(2*i-1))
                    h1 = i;
                    h2 = j;
                    which_dir_i = true; % d = e_i - e_j
                    first_linesarch_i = false;
                else
                    linesearch_i = false;
                end
            elseif (y(j) == 0e0)
                if (~skip_d(2*i))
                    h1 = j;
                    h2 = i;
                    which_dir_i = false; % d = e_j - e_i
                    first_linesarch_i = false;
                else
                    linesearch_i = false;
                end
            else % randomly choose the first direction to use
                if (rand < 5e-1)
                    if (~skip_d(2*i-1))
                        h1 = i;
                        h2 = j;
                        which_dir_i = true; % d = e_i - e_j
                        first_linesarch_i = true;
                    elseif (~skip_d(2*i))
                        h1 = j;
                        h2 = i;
                        which_dir_i = false; % d = e_j - e_i
                        first_linesarch_i = false;
                    else
                        linesearch_i = false;
                    end
                else
                    if (~skip_d(2*i))
                        h1 = j;
                        h2 = i;
                        which_dir_i = false; % d = e_j - e_i
                        first_linesarch_i = true;
                    elseif (~skip_d(2*i-1))
                        h1 = i;
                        h2 = j;
                        which_dir_i = true; % d = e_i - e_j
                        first_linesarch_i = false;
                    else
                        linesearch_i = false;
                    end
                end
            end
            
            if (linesearch_i)
                d_x = A(:,h1) - A(:,h2);
                norm_d_x = norm(d_x);
                if (norm_d_x > 0e0)
                    d_i = 1e0/norm_d_x;
                    d_x = d_x/norm_d_x;
                    ind_i_sampling = 2*i - which_dir_i;
                else
                    linesearch_i = false;
                end
            end
            
            % backtracking procedure
            while (linesearch_i && n_f<max_n_f)
                alpha_max_feas_i = norm_d_x*y(h2);
                alpha_trial = min(alpha_max_feas_i,alpha_vec(i));
                s_i = alpha_trial*d_i;
                y_trial = y;
                y_trial([h1;h2]) = [y_trial(h1)+s_i;y_trial(h2)-s_i];
                x_trial = x + alpha_trial*d_x;
                f_trial = obj(x_trial);
                n_f = n_f + 1;
                if (f_trial <= f-gamma*alpha_trial*alpha_trial)
                    expansion_i = true;
                    linesearch_i = false;
                else
                    b_sampling(ind_i_sampling) = f_trial - f;
                    alpha_sampling(ind_i_sampling) = alpha_trial;
                    ind_sampling(ind_i_sampling) = true;
                    if (theta*alpha_vec(i) <= eps_opt)
                        skip_d(ind_i_sampling) = true;
                    end
                    if (first_linesarch_i && n_f<max_n_f)
                        h3 = h1;
                        h1 = h2;
                        h2 = h3;
                        d_x = -d_x;
                        which_dir_i = ~which_dir_i;
                        ind_i_sampling = 2*i - which_dir_i;
                        first_linesarch_i = false;
                    else
                        linesearch_i = false;
                    end
                end
            end
            
            % expansion procedure
            %
            % we now produce a new point and we start a new collection
            % of samples that first includes the point where we come from
            % and the point not accepted in the expansion (if any)
            if (expansion_i)
                allow_stop = false;
                if (alpha_trial<alpha_max_feas_i && n_f<max_n_f && f_trial>f_stop)
                    y_next = y_trial;
                    x_next = x_trial;
                    f_next = f_trial;
                    f_prev = f;
                    alpha_prev = alpha_trial;
                    first_expansion = true;
                    while (expansion_i && alpha_trial<alpha_max_feas_i && n_f<max_n_f)
                        alpha_trial = min(alpha_max_feas_i,alpha_trial/delta);
                        s_i = alpha_trial*d_i;
                        y_trial([h1;h2]) = [y(h1)+s_i;y(h2)-s_i];
                        x_trial = x + alpha_trial*d_x;
                        f_trial = obj(x_trial);
                        n_f = n_f + 1;
                        if (f_trial <= f-gamma*alpha_trial*alpha_trial)
                            f_prev = f_next;
                            y_next = y_trial;
                            x_next = x_trial;
                            f_next = f_trial;
                            alpha_prev = alpha_trial;
                            first_expansion = false;
                        elseif (f_trial <= f_stop)
                            expansion_i = false;
                            y_next = y_trial;
                            x_next = x_trial;
                            f_next = f_trial;
                            alpha_prev = alpha_trial;
                            flag = 3;
                        else
                            expansion_i = false;
                            b_sampling(ind_i_sampling) = f_trial - f_next;
                            b_sampling(2*i-(~which_dir_i)) = f_prev - f_next;
                            alpha_sampling(ind_i_sampling) = alpha_trial - alpha_prev;
                            if (first_expansion)
                                alpha_sampling(2*i-(~which_dir_i)) = alpha_prev;
                            else
                                alpha_sampling(2*i-(~which_dir_i)) = (1e0-delta)*alpha_prev;
                            end
                            ind_sampling = false(2*n,1);
                            ind_sampling(ind_i_sampling) = true;
                            ind_sampling(2*i-(~which_dir_i)) = true;
                        end
                    end
                    if (~ind_sampling(ind_i_sampling)) % if this occurs, it means that the stepsize has been expanded,
                                                       % but the previous while loop ended because
                                                       % alpha_trial=alpha_max_feas_i, n_f=n_f_max or flag=3
                        b_sampling(2*i-(~which_dir_i)) = f_prev - f_next;
                        alpha_sampling(2*i-(~which_dir_i)) = (1e0-delta)*alpha_trial;
                        ind_sampling = false(2*n,1);
                        ind_sampling(2*i-(~which_dir_i)) = true;
                    end
                    y = y_next;
                    x = x_next;
                    f = f_next;
                    alpha_vec(i) = alpha_prev;
                else % if this occurs, it means that the stepsize has not been expanded
                     % because alpha_trial=alpha_max_feas_i, n_f=n_f_max or f_trial>f_stop
                    b_sampling(2*i-(~which_dir_i)) = f - f_trial;
                    alpha_sampling(2*i-(~which_dir_i)) = alpha_trial;
                    ind_sampling = false(2*n,1);
                    ind_sampling(2*i-(~which_dir_i)) = true;
                    y = y_trial;
                    x = x_trial;
                    f = f_trial;
                    alpha_vec(i) = max(alpha_trial,eps_opt);
                end
                if (f <= f_stop)
                    flag = 3;
                    break;
                end
                skip_d = false(2*n,1);
                skip_d(2*i-(~which_dir_i)) = true;
            else
                if (alpha_vec(i)> eps_opt)
                    alpha_vec(i) = max(theta*alpha_vec(i),eps_opt);
                    skip_d(ind_sampling) = false;
                    skip_d(2*i-(~which_dir_i)) = false;
                end
            end
            
        else
            
            if (alpha_vec(i)> eps_opt)
                alpha_vec(i) = max(theta*alpha_vec(i),eps_opt);
                skip_d(2*i) = false;
                skip_d(2*i-1) = false;
            end
            
        end
        
    end
    
    sampling.b = b_sampling(ind_sampling);
    v_d_temp_1 = (find(ind_sampling)+5e-1)/2e0;
    v_d_temp_2 = round(v_d_temp_1).*sign(round(v_d_temp_1)-v_d_temp_1);
    sampling.v_d = abs(v_d_temp_2).*sign(v_d_temp_2);
    sampling.alpha = max(alpha_sampling(ind_sampling),0e0);
    sampling.j = j;
    
    if (verbosity)
        fprintf('%s%i%s%.4e%s%i%s%.4e\n','it = ',it,', f = ',f,', n_f = ',n_f,', alpha_max = ',alpha_max);
        if (flag == 0)
            fprintf('%s\n','optimality condition satisfied with the desired tolerance');
        elseif (flag == 1)
            fprintf('%s\n','maximum number of function evaluations reached');
        elseif flag == 2
            fprintf('%s\n','maximum number of iterations reached');
        elseif flag == 3
            fprintf('%s\n','target objective value obtained');
        else
            fprintf('%s\n','maximum cpu time exceeded');
        end
    end
    
    t_elap = toc(t0);
    
end