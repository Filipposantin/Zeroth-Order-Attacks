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

function [x,y,f,n_f,it,t_elap,flag] = ORD(obj,A,i0,opts)
       
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
    n_atoms = size(A,2);
    if (~isnumeric(i0) || ~isreal(i0) || ~isscalar(i0) || i0~=round(i0) || i0<1 || i0>n_atoms)
        error(['the third input must be an integer between 1 and ' num2str(n_atoms)]);
    end
	
    % set options
    eps_opt = 1e-4;
    max_n_f = 100*(size(A,1)+1);
    max_it = Inf;
    f_stop = -Inf;
    max_time = Inf;
    n_atoms_in = n_atoms;
    ind_y_A = [];
    use_model = true;
    verbosity = true;
    if (nargin == 4)
        if (~isstruct(opts) || ~isscalar(opts))
            error('the fourth input (which is optional) must be a structure');
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
                    max_n_f = opts.max_n_f;
                    if (~isnumeric(max_n_f) || ~isreal(max_n_f) || ~isscalar(max_n_f) || max_n_f<=1e0)
                       error('''max_n_f'' must be a real number greater than or equal to 1');
                    end
                    max_n_f = floor(max_n_f);
                case 'max_it'
                    max_it = opts.max_it;
                    if (~isnumeric(max_it) || ~isreal(max_it) || ~isscalar(max_it) || max_it<=0e0)
                       error('''max_it'' must be a non-negative real number');
                    end
                    max_it = floor(max_it);
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
                case 'n_initial_atoms'
                    n_atoms_in = opts.n_initial_atoms;
                    if (~isnumeric(n_atoms_in) || ~isreal(n_atoms_in) || ~isscalar(n_atoms_in) || n_atoms_in<1 || n_atoms_in>n_atoms)
                       error(['''n_initial_atoms'' must be between 1 and ' num2str(n_atoms)]);
                    end
                    n_atoms_in = floor(n_atoms_in);
                case 'set_initial_atoms'
                    ind_y_A = opts.set_initial_atoms;
                    if (~isnumeric(ind_y_A) || ~isreal(ind_y_A) || ~ismatrix(ind_y_A) || any(ind_y_A<1) || any(ind_y_A>n_atoms))
                       error(['''set_initial_atoms'' must be a vector of numbers between 1 and ' num2str(n_atoms)]);
                    end
                    if (size(ind_y_A,2)>1)
                        if (size(ind_y_A,1)>1)
                            error('''set_initial_atoms'' must be a column vector');
                        end
                    elseif (size(ind_y_A,1)>1)
                        ind_y_A = ind_y_A';
                    end
                    ind_y_A = floor(ind_y_A);
                case 'use_model'
                    use_model = opts.use_model;
                    if (~islogical(use_model) && (~isscalar(use_model) || use_model<=1))
                       error('''use_model'' must be either a logical or a real number greater than 1');
                    end
                case 'verbosity'
                    verbosity = opts.verbosity;
                    if (~islogical(verbosity) || ~isscalar(verbosity))
                       error('''verbosity'' must be a logical');
                    end
                otherwise
                    error(['in the fourth input (which is optional) ''' char(opts_field(i)) ''' is not a valid field name']);
            end
        end
    end
    if (~isempty(ind_y_A))
        if (length(ind_y_A)~=n_atoms_in)
            error(['the length of ''set_initial_atoms'' must be equal to ' num2str(n_atoms_in) ' (which is the current value of ''n_initial_atoms'')']);
        end
        if (length(unique(ind_y_A)) ~= n_atoms_in)
            error('''set_initial_atoms'' contains duplicates''');
        end
        if (~(any(ind_y_A==i0)))
            error(['''set_initial_atoms'' must contain ' num2str(i0) ' (which is the index of the atom used as starting point)']);
        end
    end
	
    % In the following:
    % - A_k is a matrix with the atoms considered in the current iteration as columns
    %   (it is not a submatrix of A, since the columns of A_k may be ordered differently)
    % - n_atoms_in is the number of atoms considered in the current iteration
    %   (i.e., it is the number of columns of A_k)
    % - n_atoms_out is the number of atoms not considered in the current iteration
    % - atoms_in is a logical vector, where any component is true if the
    %   corresponding atom is considered in the current iteration, false otherwise
    % - ind_atoms_out is an integer vector containing the indices of atoms not considered in the current iteration
    % - x is the point of the current iteration in the original space
    % - y is the point of the current iteration in the transformed space
    %   (its entries are ordered accordingly to the columns of A_k, i.e., x = A_k*y)
    % - ind_y_A is a vector with the indices of the atoms corresponding to each entry of y
    %   (i.e., y(i) refers to the atom stored in the ind_y_A(i)-th column of A)
    
    if (isempty(ind_y_A))
        ind_y_A = [0 randperm(n_atoms-1)];
        ind_y_A(ind_y_A==i0) = n_atoms;
        ind_y_A(1) = i0;
        ind_y_A(n_atoms_in+1:n_atoms) = [];
    else
        ind_y_A([1 find(ind_y_A==i0)]) = [i0 ind_y_A(1)];
    end
    atoms_in = ismember(1:n_atoms,ind_y_A);
    ind_atoms_out = find(~atoms_in);
    n_atoms_out = n_atoms - n_atoms_in;
    A_k = A(:,ind_y_A);
    y = zeros(n_atoms_in,1);
    y(1) = 1e0;
    
    f = obj(A_k*y);
    
    it = 0;
    n_f = 1;
        
    f_sample_best = Inf;
    y_sample_best = sparse(n_atoms,1);
    
    flag = -1;
    sol_found_sampling = false;
    
    % line search parameters
    mu_hat = max(1e-4,eps_opt);
    gamma = 1e-6;
    theta = 5e-1; % stepsize reduction factor
    delta = 5e-1; % reciprocal of the stepsize expansion factor
    
    if (verbosity)
        fprintf('ORD starts\n')
        fprintf('-----------------------------------------------------------------------------------\n');
        fprintf('%s%s\n%s%1s\n',blanks(48),'|    local minimization details',blanks(48),'|');
        fprintf('  it         f         n_f   |A^k|     mu^k     |   n_it     n_f       eps     flag\n');
        fprintf('-----------------------------------------------------------------------------------\n');
        fprintf('%6i  %11.4e  %6i  %5i %11s  |\n',0,f,n_f,n_atoms_in,' ');
    end
    
    if (f <= f_stop)
        x = A(:,i0);
        flag = 3;
        if (verbosity)
            fprintf('%s\n','target objective value obtained');
        end
        return;
    end
    if (max_n_f <= 0e0)
        x = A(:,i0);
        flag = 1;
        if (verbosity)
            fprintf('%s\n','maximum number of function evaluations reached');
        end
        return;
    end
    if (max_it <= 0e0)
        x = A(:,i0);
        flag = 2;
        if (verbosity)
            fprintf('%s\n','maximum number of iterations reached');
        end
        return;
    end
    t_elap = toc(t0);
    if (t_elap >= max_time)
        x = A(:,i0);
        flag = 4;
        if (verbosity)
            fprintf('%s\n','maximum cpu time exceeded');
        end
        return;
    end
    
    while (true)
        
        %==================================================================
        % OPTIMIZE PHASE
        %==================================================================
        
        if (n_atoms_in > 1)
            % set DF-SIMPLEX parameters and call the solver
            if (it > 1)
                if (flag_local<=5e-1 && ~v_added && mu_hat<=eps_opt_local)
                    alpha_ini_local = eps_opt_local;
                    eps_opt_local = max(25e-2*eps_opt_local,eps_opt);
                    max_n_f_local = min(3*(n_atoms_in+1),max_n_f-n_f);
                else
                    if (flag_local <= 5e-1)
                        alpha_ini_local = eps_opt_local;
                        max_n_f_local = min(5*(n_atoms_in+1),max_n_f-n_f);
                    else
                        alpha_ini_local = max(sampling.alpha);
                        max_n_f_local = min(2*max_n_f_local,max_n_f-n_f);
                    end
                    eps_opt_local = max(75e-2*eps_opt_local,eps_opt);
                end
                df_simplex_opts.alpha_ini = alpha_ini_local;
            elseif (it == 1)
                eps_opt_local = max(5e-1,eps_opt);
                if (abs(flag_local) <= 5e-1)
                    alpha_ini_local = eps_opt_local;
                elseif (flag_local >= 5e-1)
                    alpha_ini_local = max(sampling.alpha);
                    max_n_f_local = min(10*(n_atoms_in+1),max_n_f-n_f);
                end
                df_simplex_opts.alpha_ini = alpha_ini_local;
            else
                eps_opt_local = eps_opt;
                max_n_f_local = min(2*(n_atoms_in+1),max_n_f-n_f);
            end
            df_simplex_opts.eps_opt = eps_opt_local;
            df_simplex_opts.max_n_f = max_n_f_local;
            df_simplex_opts.f_stop = f_stop;
            df_simplex_opts.f0 = f;
            df_simplex_opts.verbosity = false;
            [x,y,f,inner_n_f,inner_it,~,flag_local,sampling] = DF_SIMPLEX(obj,A_k,y,df_simplex_opts);
        else
           y = 1e0;
           inner_it = 0;
           inner_n_f = 0;
           sampling.b = [];
           sampling.v_d = [];
           sampling.alpha = [];
           flag_local = -1;
           if (it>0 && flag_local<=5e-1 && mu_hat<=eps_opt_local)
               eps_opt_local = max(25e-2*eps_opt_local,eps_opt);
           else
               eps_opt_local = max(75e-2*eps_opt_local,eps_opt);
           end
        end
        n_f = n_f + inner_n_f;
        it = it + 1;   
                
        if (flag_local == 3)
            mu = 0;
            flag = 3;
            break;
        end
        
        %==================================================================
        
        
        %==================================================================
        % REFINE PHASE
        %==================================================================
        
        v_added = false;
        mu = 0e0;
        if (n_atoms_in < n_atoms)
            
            max_norm_d_x = 0e0;
            while (~v_added && n_atoms_out>0 && n_f<max_n_f)
                
                atom_out_to_add_k = randi(n_atoms_out);
                atom_to_add_k = ind_atoms_out(atom_out_to_add_k);
                ind_atoms_out(atom_out_to_add_k) = [];
                n_atoms_out = n_atoms_out - 1;
                
                d_x = A(:,atom_to_add_k) - x;
                norm_d_x = norm(d_x);
                if (norm_d_x > 0e0)
                    max_norm_d_x = max(max_norm_d_x,norm(d_x));
                    f_trial = obj(x+mu_hat*d_x);
                    n_f = n_f + 1;
                    if (f_trial <= f - gamma*mu_hat*mu_hat)
                        mu = mu_hat;
                        f_next = f_trial;
                        if (n_f < max_n_f)
                            mu_trial = mu;
                            expansion = true;
                            while (expansion)
                                mu_trial = min(mu_trial/delta,1e0);    
                                f_trial = obj(x+mu_trial*d_x);
                                n_f = n_f + 1;
                                if (f_trial <= f - gamma*mu_trial*mu_trial)
                                    mu = mu_trial;
                                    f_next = f_trial;
                                    expansion = (mu_trial < 1e0);
                                else
                                    expansion = false;
                                    if (f_trial < f_sample_best)
                                        f_sample_best = f_trial;
                                        y_sample_best = full(sparse([ind_y_A atom_to_add_k],1,[(1e0-mu_trial)*y;mu_trial],n_atoms,1,n_atoms));
                                        if (f_sample_best <= f_stop)
                                            f = f_sample_best;
                                            y = y_sample_best;
                                            %y = abs(y)/norm(y,1);
                                            x = A*y;
                                            sol_found_sampling = true;
                                            flag = 3;
                                            break;
                                        end
                                    end
                                end
                            end
                        end
                        v_added = true;
                    elseif (f_trial < f_sample_best)
                        f_sample_best = f_trial;
                        y_sample_best = full(sparse([ind_y_A atom_to_add_k],1,[(1e0-mu_hat)*y;mu_hat],n_atoms,1,n_atoms));
                        if (f_sample_best <= f_stop)
                            f = f_sample_best;
                            y = y_sample_best;
                            %y = abs(y)/norm(y,1);
                            x = A*y;
                            sol_found_sampling = true;
                            flag = 3;
                            break;
                        end
                    end
                end
                if (flag == 3)
                    break;
                end
            end
            if (flag == 3)
                break;
            end
            
            if (~v_added)
                if (mu_hat*max_norm_d_x<=eps_opt && eps_opt_local<=eps_opt && flag_local<=5e-1)
                    flag = 0;
                    break;
                else
                    mu_hat = theta*mu_hat;
                end
            end
            
        elseif (eps_opt_local<=eps_opt && flag_local<=5e-1)
            flag = 0;
            break;
        end
        
        if (n_f>=max_n_f ||it >= max_it || (v_added && f_next<=f_stop))
            if (v_added)
                n_atoms_in = n_atoms_in + 1;
                ind_y_A(n_atoms_in) = atom_to_add_k;
                y = (1e0-mu)*y;
                y(n_atoms_in,1) = mu;
                f = f_next;
            end
            if (f <= f_stop)
                flag = 3;
            elseif (n_f >= max_n_f)
                flag = 1;
            else
                flag = 2;
            end
            break;
        end
        
        %==================================================================
        
        
        %==================================================================
        % DROP PHASE
        %==================================================================
        
        if (n_atoms_in>1 && any(y<=0e0))
            
            if ((islogical(use_model) && use_model) || (isnumeric(use_model) && (use_model>=n_atoms_in+1)))
                
                % compute the simplex gradient for the restriced problem
                % as the least-squares solution g of the linear system M*g = b
                %------------------------------------------------------------------
                
                % (1) build (part of) the matrix M from the output of DF-SIMPLEX
                %     (also (part of) the vector b is an output of DF-SIMPLEX)
                if (~isempty(sampling.alpha))
                    M = (double(repmat(1:n_atoms_in,size(sampling.v_d,1),1)==abs(sampling.v_d)))./sign(sampling.v_d);
                    M(:,sampling.j) = -1e0*sign(sampling.v_d);
                    M = (sampling.alpha).*M./vecnorm(M*A_k',2,2);
                    alpha_approx = max(sampling.alpha);
                else
                    M = [];
                    alpha_approx = 1e-6;
                end
                
                % (2) find polling samples (in the transormed space) to be added
                v_d_to_add = find(~ismember(1:n_atoms_in,abs(sampling.v_d)))';
                v_d_to_add = v_d_to_add(y(v_d_to_add)<1e0);
                if (n_f+size(v_d_to_add,1) < max_n_f)
                    if (~isempty(v_d_to_add))
                        M_to_add = [double(repmat((1:n_atoms_in),size(v_d_to_add,1),1)==v_d_to_add); -ones(1,n_atoms_in)];
                        M_to_add(:,sampling.j) = -1e0;
                        M_to_add = alpha_approx*M_to_add;
                        norm_d_x = vecnorm(M_to_add*A_k',2,2);
                        is_norm_d_x_zero = (norm_d_x<=0e0);
                        norm_d_x(is_norm_d_x_zero,:) = 1e0;
                        M_to_add = M_to_add./norm_d_x;
                    else
                        M_to_add = -alpha_approx*ones(1,n_atoms_in);
                        norm_d_x = norm(M_to_add*A_k');
                        if (norm_d_x > 0e0)
                            M_to_add = M_to_add/norm_d_x;
                            is_norm_d_x_zero = false;
                        else
                            is_norm_d_x_zero = true;
                        end
                    end
                    
                    % (3) compute the objective function at the new points and add these values to b
                    size_M_to_add = size(M_to_add,1);
                    b_to_add = zeros(size_M_to_add,1);
                    for i = 1:size_M_to_add
                        if (~is_norm_d_x_zero(i))
                            b_to_add(i) = obj(x+A_k*(M_to_add(i,:)'))-f;
                            n_f = n_f + 1;
                        end
                    end
                    M = [M; M_to_add];
                    sampling.b = [sampling.b; b_to_add];
                    
                    try
                        
                        % (4) compute the simplex gradient
                        g_approx = pinv(M'*M)*(M'*sampling.b);
                        
                        %------------------------------------------------------------------
                        
                        gx = g_approx'*y; % approximation of gradient-vector product
                        
                        % (5) find atoms to remove
                        ind_atoms_to_remove = find(y<=0e0 & gx*ones(n_atoms_in,1)<=g_approx);
                        
                    catch ME
                        
                        % something went wrong when computing the simplex gradient
                        warning('An error occured when computing the simplex gradient %s,\n%s\n', ...
                            ['(' ME.identifier, ')'], 'Simplex gradient will not be used in this iteration.');
                        ind_atoms_to_remove = find(y<=0e0);
                    
                    end
                
                elseif (n_f < max_n_f)
                    % not enough function evaluations at our disposal to compute the simplex gradient
                    ind_atoms_to_remove = find(y<=0e0);
                else
                    % the maximum number of function evaluations is reached
                    if (v_added)
                        %y = abs(y)/norm(y,1);
                        x = A_k*y;
                    end
                    flag = 1;
                    break;
                end                
            else
                ind_atoms_to_remove = find(y<=0e0);
            end
        
        else
            ind_atoms_to_remove = [];
        end
        
        % add atoms
        if (v_added)
            atoms_in(atom_to_add_k) = true;
            n_atoms_in = n_atoms_in + 1;
            A_k(:,n_atoms_in) = A(:,atom_to_add_k);
            ind_y_A(n_atoms_in) = atom_to_add_k;
            y = (1e0-mu)*y;
            y(n_atoms_in,1) = mu;
            f = f_next;
        end
        
        % remove atoms
        n_atoms_to_remove = length(ind_atoms_to_remove);
        if (n_atoms_to_remove > 0)
            atoms_in(ind_y_A(ind_atoms_to_remove)) = false;
            A_k(:,ind_atoms_to_remove) = [];
            y(ind_atoms_to_remove) = [];
            ind_y_A(ind_atoms_to_remove) = [];
            n_atoms_in = n_atoms_in - n_atoms_to_remove;
        end
        
        ind_atoms_out = find(~atoms_in);
        n_atoms_out = n_atoms - n_atoms_in;
        
        %==================================================================
        
        t_elap = toc(t0);
        if (t_elap >= max_time)
            flag = 4;
            break;
        end
        
        % iteration prints
        if (verbosity)
            fprintf('%6i  %11.4e  %6i  %5i  %10.4e  |  %6i  %6i  %10.4e  %3i\n',it,f,n_f,n_atoms_in,mu,inner_it,inner_n_f,eps_opt_local,flag_local);
        end
                
    end
    
    % build x from y
    if (~sol_found_sampling)
        if (f_sample_best < f)
            f = f_sample_best;
            y = y_sample_best;
            %y = abs(y)/norm(y,1);
            x = A*y;
        else
            y_temp = y;
            y = zeros(n_atoms,1);
            y(ind_y_A) = y_temp;
        end
    end
    
    % final iteration prints
    if (verbosity)
        fprintf('%6i  %11.4e  %6i  %5i  %10.4e  |  %6i  %6i  %10.4e  %3i\n',it,f,n_f,n_atoms_in,mu,inner_it,inner_n_f,eps_opt_local,flag_local);
        if (flag == 0)
            fprintf('%s\n','optimality condition satisfied with the desired tolerance');
        elseif (flag == 1)
            fprintf('%s\n','maximum number of function evaluations reached');
        elseif (flag == 2)
            fprintf('%s\n','maximum number of iterations reached');
        elseif (flag == 3)
            fprintf('%s\n','target objective value obtained');
        else
            fprintf('%s\n','maximum cpu time exceeded');
        end
    end
    
    t_elap = toc(t0);
    
end