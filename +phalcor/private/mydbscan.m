function [c, neighbors] = mydbscan(X, isNeighbor, minPts)
% Author: Tom Shlomo, ACLab BGU, 2020

n = size(X,1);
c = nan(n,1);
neighbors = zeros(n,1);
C = 0;
for i=1:n
    if ~isnan(c(i))
        continue
    end
    queue = zeros(n,1);
    neighbors_logical = isNeighbor(X(i,:), X);
    neighbors(i) = nnz(neighbors_logical);
    queue(1:neighbors(i)) = find(neighbors_logical);
    if neighbors(i) < minPts
        c(i) = 0;
        continue
    end
    C = C+1;
    c(i) = C;
    queue_index = 0;
    queue_size = neighbors(i);
    while queue_index+1<=n && queue(queue_index+1)>0
        queue_index = queue_index+1;
        k = queue(queue_index);
        if k==i
            continue
        end
        if c(k)==0
            c(k) = C;
        end
        if ~isnan(c(k))
            continue
        end
        c(k) = C;
        neighbors_logical = isNeighbor(X(k,:), X);
        neighbors(k) = nnz(neighbors_logical);
        if neighbors(k) >= minPts
            newNeighbors = setdiff(find(neighbors_logical), queue(1:queue_size));
            queue(queue_size+1:queue_size+length(newNeighbors)) = newNeighbors;
            queue_size = queue_size + length(newNeighbors);
        end
    end

end

end

