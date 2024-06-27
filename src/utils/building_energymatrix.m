function [E, info, output, partition] = building_energymatrix(input)
% [E, info, output, partition] = building_energymatrix(input)
% reading data from the input and get the corresponding energy matrix
%
% INPUT: 
%       input - output data from Chimera 
%               k by 5 matrix, where k is the number of edges.
% OUTPUT:
%       info - data structure:
%           info.k = number of edges
%           info.n0= number of nodes
%           info.p_all = number of partitions
%           info.p = number of nonempty partitions
%           info.partition_ind = indices of partitions that are nonempty.
%           info.m_all = p_all-(column) vector (including empty paritions)
%           info.m = p-(column) vector [m_1; m_2; ...; m_p]

k = size(input, 1); %gets number of lines in file (number of edges)
info.k = k;


[partition, ~, n_partition] = ...
    unique([input(:,[1,2]); input(:,[3,4])],'rows');
partition_extra = zeros(length(partition),1);

% partition is a vector containing all the nodes in the graph without
% repetition
% each node is defined by a pair of numbers. first number gives the
% partition it belongs to. Second number identifies it within that
% partition. 

%finds largest partition number and adds one

p_all = max(partition(:,1))+1; 
%I don't think we should add one since the numbering starts from 1 not 0
info.p_all = p_all; %number of partitions

m_all = zeros(p_all,1); %zero vector of size equal to number of partitions

for i = 1:p_all
    ind = logical(partition(:,1) == i-1);
    m_all(i) = sum(ind); % save number of occurences of partition number i
    if m_all(i)>0
        partition_extra(ind) = 0:m_all(i)-1; 
    end
end

%keyboard

clear ind;

partition = [partition, partition_extra];
clear partition_extra;

info.m_all = m_all;
info.partition_ind = find(m_all~=0); % disregard empty partitions
info.m = m_all(logical(m_all~=0)); % m_i gives the size of partition i
info.p = nnz(m_all); %the final number of partitions
info.n0 = sum(m_all); %sum of m_i



output = zeros(info.k, 7); %one row for each edge
output(:, [1:4, 7]) = input; %fill in columns 1,2,3,4 and 7

E = spalloc(info.n0, info.n0, 2* info.k);

for i = 1:info.k
    % first we need to replace 
    % input(i,1:2) and input (i,3:4)
    % by the proper nodes.
    
    input(i,1:2) = partition(n_partition(i),[1,3]);
    input(i,3:4) = partition(n_partition(info.k + i),[1,3]);
    
    n1 = sum( m_all(1:input(i,1)) ) + input(i,2) + 1;
    n2 = sum( m_all(1:input(i,3)) ) + input(i,4) + 1;
    output(i, 5) = n1;
    output(i, 6) = n2;
    
    %building E
    E(n1, n2) = input(i, 5);
    E(n2, n1) = input(i, 5);
end

