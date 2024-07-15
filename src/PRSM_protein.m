function [Y, Out] = PRSM_protein(Eorig, V, J, opts)
%function [Y, Out] = PRSM_protein(Eorig, V, J, opts)
% algorithm for solving DNN relaxation of the side-chian positioning problem
% (DNN)   min    trace(Eorig*Y)
%         s.t.   Y = V*R*V^T
%                R in {R : trace(R) = p+1, R positive semidefinite}
%                Y in {Y : G_J(Y) = E_{00}, 0<=Y<=1}
%INPUT : Eorig : objective functiton data, symmetric matrix
%        V     : facial range vector
%        J     : gangster indices
%        opts  : options, a strcut
%                .R0,Y0,Z0 : initial iterates
%                .m : vector holding the size of each rotamer set
%                .tol : stopping tolerance
%                .cal_bd : binary value, compute lower and upper bound
%                .gamma : under-relaxation parameter
%                .beta : penalty parameter for augmented Lagrangian 
%                .maxiter : max number of allowed iterations
%OUTPUT:  
%   Y   : variable Y
%   Out : a strct, 
%          .xbest: best upper bound found
%          .ubd and .lbd : history of upper and lower bounds
%          .toc: time taken
%          .iter: number if iterations
%          .obj: history of objective value trace(Eorig*Y)
%          .pr and .dr: history of primal (pr) and dual (dr) residual
%          .Z: dual multiplier
%          .relgap: relative gap computed by best lower and upper bounds
%          .ubest and .lbest: best upper and lower bounds

%% parameter setting
if isfield(opts,'maxit')     maxit = opts.maxit;       else maxit = 500;    end
if isfield(opts,'tol')       tol = opts.tol;           else tol = 1e-1;     end
if isfield(opts,'metol')     metol = opts.tol;      else metol = 1e-10;     end
if isfield(opts,'beta')      beta = opts.beta;         else beta = 50;      end
if isfield(opts,'gamma')     gamma = opts.gamma;       else gamma = 1;      end
if isfield(opts,'cal_bd')    cal_bd = opts.cal_bd;     else cal_bd = 1;     end
if isfield(opts,'nstop')    nstop = opts.nstop;     else nstop = 100;     end % norm stalling condition


m = opts.m;
n0 = sum(m);
p = length(opts.m);
msum = cumsum(m);
msumsh = [0;msum(1:end-1)];


% Forming the matrix A; represents one rotamer is chosen from each set
A = [];
for j = 1:p  
    A = blkdiag(A,ones(1,m(j)));
end

% initial values for R,Y,Z
R0 = opts.R0;
Y0 = opts.Y0;
Z0 = opts.Z0;
R = R0; Y = Y0; Z = Z0;
nrm_pR=1;
nrm_dR=1;

Y( (abs(Y)<min([nrm_pR,nrm_dR,1e-9])) ) = 0; %removing small values
Z( (abs(Z)<min([nrm_pR,nrm_dR,1e-9])) ) = 0; %removing small values

E = Eorig;

% detect collision
N_E = full(sum( E(E<0))); % sum of negative values of the energy matrix; natural lower bound to the optimal value
xfeas = zeros(n0,1);
xchop = mat2cell(rand(1,n0),1,m)'; %chopped Vcol into appropriate sizes
[~,r] = cellfun(@max,xchop); %apply max function to each cell array
xfeas( msumsh+r ) = 1; %placing 1 at the max position at each partition
Out.xbest = xfeas;
xfeas = [1;xfeas];
NewThreshold = find( Eorig > xfeas'*(Eorig*xfeas) - N_E ); % these entries can be added to the gangster indices

Ezeros = zeros(n0+1);
Ezeros(NewThreshold) = 1;

%%% diagonal gangster -> whole row and column is gangster
diagone = find(diag(Ezeros)==1);

if ~isempty(diagone)
   keyboard 
    Ezeros(:,diagone) = 1;
    Ezeros(diagone,:) = 1;
   fprintf('\t %d diagonal gangster indices found\n',length(diagone))
end

J0 = J;
Jnew = logical( logical(J) + logical(Ezeros) ) ; % add the new ganster indices 
J = Jnew;   % update the gangster
E(J) = 0;

fprintf('\t number of new Gangster indices added =%d \n',nnz(xor(J0,J)))



%%% indices for projecting the arrow positions of Z
%%% we use these for dual update projection
Zdiaginds = speye(n0+1);
Zdiaginds(1,1) = 0;
temp = [0        ones(1,n0)
    ones(n0,1)     spalloc(n0,n0,0)];
Zdc = logical(Zdiaginds + sparse(temp));
dcE = -E(Zdc);  %%% vectors to use for updating Z

Z(Zdc) = dcE; % This is the initial Z iterate of PRSM


%%% saving output
obj = zeros(maxit,1);
hist_pR = zeros(maxit,1);
hist_dR = zeros(maxit,1);

nstall = 0;   % norm error measures do not change
ustall = 0;   % upper bound measures do not change
lstall = 0;   % lower bound measures do not change
iter = 0;

% setting parameters for stopping criteria
lbest = full(sum( E(E<0)));
ubest = xfeas'*(Eorig*xfeas);

Out.ubd = [ubest];
Out.lbd = [lbest];

% start_time = tic;   % start time  %% wall-clock
start_time = cputime;   % start time  cputime!


fprintf('\t iter# \t     nrm_pR \t \t   nrm_dR \t \t   nrm_dZ \t \t  rel.gap \t \t  obj.val  \t \t    time \t \t  \n');
fprintf('\t _____ \t   __________\t \t __________\t \t  _________ \t  _________ \t __________  \t __________  \n')

start_iter_time = tic;

V=full(V);

while  (nstall < nstop) &&  (iter < maxit) && (ubest-metol > lbest)
    iter = iter + 1;
    
    %% STEP 1: R update -  argmin_{R in set R} ||R - V^T*(R+Z/beta)*V ||
    fW = (Y + (Z/beta));
    WV = V'*(fW*V);
    WV = (WV'+WV)/2;  % symmetrize for eig or difficulties arise
    WV = full(WV);    % WV = V^T*(R+Z/beta)*V
    
    [U,S] = eig(WV);
    % project WV to the set R
    Sdiag=diag(S);
    Rproj = simplex_proj(Sdiag,p+1);
    
    id = find(Rproj>0);   % find the positions of pos. eigs
    sn=length(id);
    
    if sn==size(WV,1)
        VRV=V*WV*V';
    else
        if ~isempty(id)
            tempid = V*U(:,id);
            VRV = tempid*(spdiags(Rproj(id),0,sn,sn)*tempid');%Eckert-Young projection
        else
            VRV = zeros(length(V));
        end
    end
    
    %% STEP 2: Z update (first dual update)   
    Z = Z + gamma*beta*(Y-VRV);
    % projection arrow positions onto the known dual optimal multiplers
    Z(Zdc) = dcE;  
    Z=(Z+Z')/2;
    Z( (abs(Z) < min([nrm_pR,nrm_dR,1e-9])) ) = 0;
        
    
    %% STEP 3: Y update - argmin_{Y in set Y} ||Y - (VRV^T - (E+Z)/beta)||
    %%%        projection onto the gangster constraint and 0<=Y<=1, 
    Y = VRV-(E+Z)/beta;
    
    %%% projection onto 0<=Y<=1
    Y = min(1, max(0,Y));
    %%% projection onto gangster
    Y(J) = 0; Y(1,1) = 1;
    
    Y=(Y+Y')/2;
    Y( (abs(Y) < min([nrm_pR,nrm_dR,1e-9])) ) = 0; % removing small entries - round to 0
      
    pR = Y-VRV;
    dR = Y-Y0; Y0 = Y;
    
    if any(Z < min(diag(Z) ) +0 )
        keyboard   % just to see if Z becomes very small (but this should not happen)
    end
    
    %% STEP 4: Z update (second dual update)
    Z = Z + gamma*beta*pR;
    % projection arrow positions onto the known dual optimal multiplers
    Z(Zdc) = dcE; 
    
    Z=(Z+Z')/2;
    Z( (abs(Z) < min([nrm_pR,nrm_dR,1e-9])) ) = 0;
    
    nrm_dZ = norm(Z-Z0,'fro');
    Z0 = Z;
    
    
    %% Iterate Summary
    obj(iter) = sum(E(:).*Y(:));  % objective function value at current iterate
    nrm_pR = norm(pR,'fro');      % primal residual
    nrm_dR = beta*norm(dR,'fro'); % dual residual
    hist_pR(iter) = nrm_pR;       % save primal residual history
    hist_dR(iter) = nrm_dR;       % save dual residual history

    if nrm_pR < 1e0 && nrm_dR < 1e0
        if cal_bd && mod(iter,100) == 0  
            %% find lbd, ubd in Out.lbd, Out.ubd
            cal_lbd;   % call cal_lbd funciton below
            cal_ubd;   % call cal_ubd funciton below
                        
            
            %% strategy to add more gangster indices (find collision)
            xfeas = [1;Out.xbest]; 
            NewThreshold = find( Eorig > xfeas'*(Eorig*xfeas) - N_E );     
                        
            Ezeros = zeros(n0+1);
            Ezeros(NewThreshold) = 1;
            
            %%% diagonal gangster -> whole row and column is gangster
            diagone = find(diag(Ezeros)==1); % find diagonal gangster
            if ~isempty(diagone)
                %keyboard
                Ezeros(:,diagone) = 1;  % add column gangster indices
                Ezeros(diagone,:) = 1;  % add row gangster indices
                fprintf('\t d diagonal gangster indices found\n',length(diagone))
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            J0 = J;  % save the old gangster
            Jnew = logical( logical(J) + logical(Ezeros) ) ; %update the new gangster set
                        
            J = Jnew;
            E(J) = 0;
            if nnz(J) ~= nnz(J0)
                fprintf('\t number of new Gangster indices added = %d \n',nnz(J)-nnz(J0));
            end
            
        end
        
    end
    %% checking stopping condition
    if nrm_pR < tol && nrm_dR < tol
        nstall = nstall + 1;
    else
        nstall = 0;  % initialize the count
    end
    
    if mod(iter,100) == 0
        ubest = min( Out.ubd);
        lbest = max( Out.lbd);
        relgap = 2*(ubest-lbest)/abs(ubest+lbest+1);
        
        fprintf('\t %i \t\t%5.4e \t \t%5.4e \t \t %5.4e\t \t %5.4e \t%5.4e \t  %5.4e\n',...
            iter,nrm_pR,nrm_dR,nrm_dZ,relgap,obj(iter),toc(start_iter_time));  
             
        
    end
    
    
end % of main iteration ... end of while loop


% Final update of the iterates 
%% R update    to get proper pR at the end
fW = (Y + (Z/beta));
WV = V'*(fW*V);
WV = (WV'+WV)/2;  % symmetrize for eig or difficulties arise
WV = full(WV);


[U,S] = eig(WV);
Sdiag=diag(S);
Rproj = simplex_proj(Sdiag,p+1);

id = find(Rproj>0);   % find the positions of pos. eigs
sn=length(id);

if sn==size(WV,1)
    VRV=V*WV*V';
else
    if ~isempty(id)
        tempid = V*U(:,id);
        VRV = tempid*(spdiags(Rproj(id),0,sn,sn)*tempid');%Eckert-Young projection
    else
        VRV = zeros(length(V));
    end
end
pR = Y-VRV;
nrm_pR = norm(pR,'fro');
Z = Z + gamma*beta*(Y-VRV);
Z = (Z+Z')/2;

Yround = round(Y);
Ycheck = zeros(size(Y));
Ycheck(J) = Yround(J);
if norm(Ycheck,'fro') >1e-14
    keyboard
end

%%%% call special  versions of the bounding for extra accuracy
% The last call to uppper/lower bounds
cal_ubd;
cal_lbd;
mubest = min(full(Out.ubd));
mlbest = max(Out.lbd);
relgap = 2*abs(mubest-mlbest)/( abs(mubest)+abs(mlbest)+1);
fprintf('\t %i \t\t%5.4e \t \t%5.4e \t \t %5.4e\t \t %5.4e \t%5.4e \t  %5.4e\n',...
    iter,nrm_pR,nrm_dR,nrm_dZ,relgap,obj(iter),toc(start_iter_time));

if iter >= maxit
    fprintf('\t It exceeded the max number of iterations.\n')
end


%%% save output
Out.toc = cputime-start_time;  % compute the cputime
Out.obj = obj(1:iter);
Out.iter = iter;
Out.pr = hist_pR(1:iter);
Out.dr = hist_dR(1:iter);
Out.Z = Z;
Out.Etrunc = E; % truncate huge numbers
Out.relgap = relgap;
Out.ubest = mubest;
Out.lbest = mlbest;


% print the final interate information after the while loop
fprintf('\n \t at end of while loop :\n')
fprintf(' \t nstall = %i,ustall = %i,lstall = %i,iter = %i,[lbest ubest] = [%g %g]\n',...
    nstall,ustall,lstall,iter,mlbest,mubest)




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function cal_lbd     % calculate lower bound
        %lbd = min_{Y \in setY} <E+Z,Y> + min_{R in setR} <-VZV,R>
        
        %%%%%%% STEP 1 : compute min_{R in setR} <-VZV,R> 
        %%%%%%%                  i.e., sum w.r.t. R in R  
        VZV = V'*Z*V;
        VZV = (VZV+VZV')/2;
        Zeigmax = max( eig(VZV)); % compute max(eig(VZV))
    
        %%%%%%% STEP 2 : copute  min_{Y \in setY} <E+Z,Y> 
        %%%%%%%                  i.e., sum w.r.t. Y in Y  
        EZ = E+Z;
        %%% constraint 0<=Y<=1 %%%
        temp = EZ(1,1); %save to add to bound and zero out for now
        EZ(Zdc) = 0;   % these are zero due to projection of Z onto arrow of E
        EZ(1,1) = 0;   % (0,0) corner of the gangster index
        EZ(J) = 0;
    
        lbd = sum(sum(EZ(E<-Z)))+ temp -  (p+1)*Zeigmax; 
        
        %%% computing/saving lower bound
        Out.lbd = [Out.lbd; lbd];
        
    end  % end of  'function cal_lbd'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function cal_ubd        % calculate upper bound
        % Two strategies are used for projection onto the feasible set
        % Projections are performed without using an external sovler
        % 1. use the first columns of Y
        % 2. use the most dominant eigen vector of Y
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute feas_obj1 : use the first column of Y
        xapprox = zeros(n0,1);
        Ycol = Y([2:end],1);
        Ychop = mat2cell(Ycol',1,m)'; % chopped Vcol into appropriate sizes
        [~,d] = cellfun(@max,Ychop);  % apply max function to each cell array
        xapprox( msumsh+d ) = 1;  % placing 1 at the max position at each partition
        
        xapprox = logical([1;xapprox]);
        if sum(xapprox)~= p+1  % extra check
            error('xapprox is not feasible \n')
        end
        
        %%% x'Ex = trace Exx' = trace EX = sum(E(X(:));
        Xapprox = logical(xapprox*xapprox');
        feas_obj1 = sum(Eorig(Xapprox(:)));
        
        if feas_obj1 <= min(Out.ubd) 
            Out.xbest = xapprox(2:end);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %feas_obj2 : pick the most dominant eigenvector of Y
        xapprox = zeros(n0,1);
        [u,d] = eigs(Y,1,'LA');
        Ydom = sqrt(d)*u(2:end);  % obtain the most dominant eigenvector
        Ychop = mat2cell(Ydom',1,m)'; %chopped Vcol into appropriate sizes
        [~,d] = cellfun(@max,Ychop); %apply max function to each cell array
        xapprox( msumsh+d ) = 1; %placing 1 at the max position at each partition
        
        xapprox = [1;xapprox];
        Xapprox = logical(xapprox*xapprox');
        if sum(xapprox)~= p+1
            error('xapprox is not feasible \n')
        end
        
        feas_obj2 =  sum(Eorig(Xapprox(:)));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        if feas_obj2 <= min(Out.ubd) 
            Out.xbest = xapprox(2:end);
        end
     
        ubd_curr = min([feas_obj1,feas_obj2]); 
        
        Out.ubd = [Out.ubd;  ubd_curr];       
    end   % end of 'function cal_ubd'  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



end   % of main function PRSM_protein


