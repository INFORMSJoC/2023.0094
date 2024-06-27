%%% This is a script for creating tables (in .tex) for selected instances
%%% fron PDB. 
%%% Select instances by chaning the array values probnumbers.
%%% The path holding the instnaces must be specified by the user; otherwise
%%% the code will not run. Specify the path to the variable 'opts.foldername'.

%%% The overview of the routine:
%%% 1) this script: specify the instance numbers
%%% 2) run_tests : construct the problem data for each instance, make numerical report in .tex format
%%% 3) PRSM_protein : runs algorithm

% Initialize
clear

seed = 100;
rng(seed);

% path to the solver
addpath(genpath('.\src'));

% Select test instances; integer array from 1 to 131
probnumbers = [2,6,8];     

% probnumbers = [1:99] consists of instances up to 99 amino acid
%             = [41:95] consists of instances with 100~199 amino acids  
%             = [96:131] consists of instances with 200-299 amino acids


% path to the xxxxData.txt files. This must be specified to 
% complete the routine run_tests(probnumbers, filename, opts)
datafolder = ".\data";  

tablefilename = 'tabletest.tex';

% parameters for solver 
opts = []; 
opts.gamma = 0.99;  % under-relaxation parameter 
opts.tol = 1e-10;   % epsilon in the paper 
opts.nstop = 100;   % s_t in the paper; max sequential stalling allowed 

%% Run tests
run_tests(probnumbers, tablefilename, datafolder, opts);
