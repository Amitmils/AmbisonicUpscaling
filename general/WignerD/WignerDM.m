function DM=WignerDM(N,alpha,beta,gamma)
DM=zeros((N+1)^2);
    for n=0:N
        for m=-n:n
            DM(n^2+n+m+1,n^2+1:(n+1)^2)=...
                wignerd_new(n,m,alpha,beta,gamma);
        end
    end
end