function run_tests(probnumbers, filename, foldername, opts)
%function run_tests(probnumbers, filename, foldername)

%%% This sets up options, calls/loads the correct problem
%%% calls the solver PRSM; and outputs the final table in latex format.

if isfield(opts,'tol')     tol = opts.tol;        else tol = 1e-10;   end 
if isfield(opts,'gamma')   gamma = opts.gamma;    else gamma = 0.99;  end % under-relaxation parameter
if isfield(opts,'nstop')   nstop = opts.nstop;    else nstop = 100;   end % norm stalling condition

% arrays for saving the problem information 
numProbs = length(probnumbers);
plist = zeros(numProbs,1);
n0list = zeros(numProbs,1);
% arrays for saving the algorithm outputs 
bestlbds = zeros(numProbs,1);
bestubds = zeros(numProbs,1);
relgaps = zeros(numProbs,1);
iters = zeros(numProbs,1);
timessec = zeros(numProbs,1);

f = fopen(strcat(foldername,'\131ProteinList.txt'));
datalist = textscan(f,'%s');
fclose(f);
datalist = string(datalist{:});  % list containing instance names
rownumb=0;


%% Choose test data set
for i=probnumbers
    rownumb = rownumb+1;  % row number in table
    
    problemname = datalist(i);  % pick the i-th instance from the list
    
    
    file = fullfile(foldername,strcat(problemname,'data.txt'));
    input = load(file);   % loads the data matrix
    
    input(:, 1) = [];     % removes the first column of indices 1:#rows
    
    fprintf(2,'Impoting instance #%d %s\n',i,problemname)
    
    
    % build the energy matrix
    [E, info, output, partition] = building_energymatrix(input);
    
    
    % multiplying 0.5 to the off-diagonal elements for the accurate energy
    % value
    diagE = diag(E);
    E = 0.5*E;
    E(eye(length(E))==1) = diagE;  % do not scale the diagonal entries
    
    m = info.m;  % array holding the number of rotamers for each residue
    n0 = sum(m); % number of total vertices (rotamers)
    p = info.p;  % number of residues
    
    
    fprintf('# of rotamers n_0 = %d, # of amino acids p = %d\n',n0,p)
    
    msum = cumsum(m);
    msumsh = [0;msum(1:end-1)];
    E = blkdiag(0,E);
    
    
    % set algorithm patameter
    beta =  round(0.5*n0/p);
    beta = max(beta,1);
    maxit = min(1e5,(p*(n0+1))+10000);  
    
    %% obtain farial range vector V
    Vorig = building_V(m);
    [Q,~] = qr(Vorig);
    V = Q(:,1:n0+1-p);
    
    testorth = V'*V; %check that Vhat'*Vhat = I
    errornormVhat = norm(full((testorth-speye(length(testorth)))));
    fprintf('error in ||V''V-I|| is = %g; sparsity Vhat is nnz/numel = %g \n',...
        errornormVhat,nnz(V)/numel(V));
    
    %% Forming the gangster index set J
    C = cell(p,1);
    for j = 1:p  %I can't vectorize this computation
        C{j} = triu(ones(m(j)),1)/m(j);
    end
    C = blkdiag(C{:});
    JC = C+C';
    JC = sparse(JC~=0);
    
    J = sparse( blkdiag(0,JC) );
    J = sparse(J~=0);
    
    
        
    
    
    %% set initial points and parameters
    Y0 = zeros(n0+1);
    Z0 = zeros(n0+1);  % the initial will be changed in the algorithm to hold arrow proj of Z
    
    
    %% Set/pass options to the solver
    opts.R0 =  V'*Y0*V; opts.Y0 = Y0; opts.Z0 = Z0;  % initial iterates
    opts.m = m;
    opts.maxit = maxit;    
    opts.beta = beta;
    opts.gamma = gamma;    % under-relaxation parameter
    opts.tol = tol;        % more accuracy for high rank
    opts.metol = 1e-10;    % tolerance for lower and upper bound computation
    opts.nstop = nstop;    % norm stalling condition
    opts.cal_bd = 1;       % calculate upper and lower bounds
    fprintf('max # iterations = %i; tolerance = %g \n', opts.maxit,opts.tol)
    fprintf('penalty paramter beta = %g, under-relaxation paramter gamma = %g \n',...
        beta,opts.gamma);
    fprintf('\n\n\t <strong> [[ calling ''PRSM protein'' ]]</strong>\n\n');
    
    
    %% Call the solver
    [Y,Out] = PRSM_protein(E,V,J,opts);
    %% Report the output
    hms = sec2hms(Out.toc);
    fprintf('\t total toc/sec time %g \n',Out.toc);
    
    

    fprintf('\nFrom run_tests:\n Max.Lbd.Val: %g\n Min.Ubd(Feas.)Val: %g\n #Iters : %i\n CPU: %s \n Rel.Gap: %g \n\n',...
        Out.lbest,Out.ubest, Out.iter,hms,Out.relgap);
        
    
    bestlbds(rownumb) = Out.lbest;
    bestubds(rownumb) = Out.ubest;
    relgaps(rownumb) = Out.relgap;
    iters(rownumb) = Out.iter;
    timessec(rownumb) = Out.toc;
    plist(rownumb) = p;
    n0list(rownumb) = n0;   


    
end  % of for loop


%% Output .tex table
fmt1 = '%i  & %s  & %i & %i & %.5f & %.5f & %.5e & %i & %8.2f  ';
fmt3 = '%7.2f & %7.2f & %7.2f & %7.2f & %7.2e & %7.2e';
fmt4 = ' \\cr\\hline\n';
fmt5 = ' \\cr \n';

if isfile(filename)
    system(['del ', filename]);
end
fid = fopen(filename, 'w');

fprintf(fid,'\\documentclass{article} \n');
fprintf(fid,'\\usepackage{multicol} \n');
fprintf(fid,'\\usepackage{bm} \n');
fprintf(fid,'\\begin{document} \n');


fprintf(fid, '%s\n', '\begin{tabular}{|cccc||ccc||cc|} \hline');
fprintf(fid, '%s', [...
    '\multicolumn{4}{|c||}{\textbf{Problem Data}} & ', ...
    '\multicolumn{3}{|c||}{\textbf{Numerical Results}} & ', ...
    '\multicolumn{2}{|c|}{\textbf{Timing}} ']);
fprintf(fid, fmt4);
fprintf(fid, '%s', [...
    '\multicolumn{1}{|c|}{\#} & ', ...
    '\multicolumn{1}{|c|}{\textbf{name}} & ', ...
    '\multicolumn{1}{|c|}{$\bm{p}$} & ', ...
    '\multicolumn{1}{|c||}{$\bm{n_0}$} & ', ...
    '\multicolumn{1}{|c|}{\textbf{lbd}} & ', ...
    '\multicolumn{1}{|c|}{\textbf{ubd}} & ', ...
    '\multicolumn{1}{|c||}{\textbf{rel-gap}} & ', ...
    '\multicolumn{1}{|c|}{\textbf{iter}} & ', ...
    '\multicolumn{1}{|c|}{\textbf{time(sec)}} ']);
fprintf(fid, fmt4);
jj = 1;

fprintf(fid, fmt1, probnumbers(jj), datalist(probnumbers(jj)),...
    plist(jj), n0list(jj), ...
    bestlbds(jj), bestubds(jj), relgaps(jj), ...
    iters(jj), timessec(jj)  );

for jj = 2:numProbs
    fprintf(fid, fmt5);
    fprintf(fid, fmt1, probnumbers(jj), datalist(probnumbers(jj)),...
        plist(jj), n0list(jj), ...
        bestlbds(jj), bestubds(jj), relgaps(jj), ...
        iters(jj), timessec(jj));
end
fprintf(fid, fmt4);
fprintf(fid, '\\end{tabular}\n');

fprintf(fid,'\\end{document} \n');
fclose(fid);


% print out the texts in .tex file to command window
if ismac     % Mac platform
    system(['echo ', filename]);
elseif isunix  % Linux platform
    system(['cat ', filename]);
elseif ispc    % Windows platform
    system(['type ', filename]);
else
    disp('output platform not supported')
end



