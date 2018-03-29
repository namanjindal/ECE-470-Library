function T = find_fk(S, theta, M)
% Returns a pose T
% Given a set of n spatial screw axis S = [S1 S2 .. Sn]
% a matrix of n thetas, where theta can be 1xn or nx1
% and the current pose, M
    % n is number of joints
    [m,n] = size(S);
    T = 1;
    for N = 1:n
        T = T * expm(skew4(S(:,N))*theta(N));
    end
    T = T * M;
end
