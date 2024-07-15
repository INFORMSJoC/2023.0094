function W = building_V(m)
% function W = building_V(m)
% produces a full-column rank matrix whose column space is the
% null space of Aplambda.
% assumes each entry of m is greater than 1.
% input: 
%       m - p-vector consisting of partition size 
%          (each entry of m is at least 2)
%       p - number of partitions
% output:
%       W==V - a full-column rank matrix whose column space is the
%              null space of Aplambda.

p = length(m);
if size(m,1) ~= 1
    if size(m,2) ~= 1
        error('m must be a vector.');
    end
    m = m';
end
n0 = sum(m);
num_nonzeros = 1+p + 2*(n0-p);
mcumsum = cumsum(m)+1;
row_ind = zeros(1,num_nonzeros);
rpointer = 0;
for ii=1:p
    if m(ii)~=0 % safety check
        row_ind(rpointer+1:rpointer+m(ii)-1) = ...
            rpointer+ii+1:rpointer+m(ii)+ii-1;
        row_ind(rpointer+n0-p+1:rpointer+m(ii)-1+n0-p) = ...
            ones(1, m(ii)-1)*mcumsum(ii);
        rpointer = rpointer+m(ii)-1;
    end
end
row_ind(2*(n0-p)+1: end) = [1, mcumsum];
col_ind = [2:n0-p+1, 2:n0-p+1, ones(1,p+1)];
entries = [ones(n0-p,1); -ones(n0-p,1); ones(p+1, 1)];
W = sparse(row_ind, col_ind, entries, n0+1, n0-p+1);
