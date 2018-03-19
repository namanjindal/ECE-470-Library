function J = jacobian(n, S, theta)
% Finds the space Jacobian of an 6xn matrix
% Takes in the number of joints of the robot, n
% a matrix of n spatial screw axis S = [S1 S2 ... Sn]
% and a matrix of n thetas
    J = zeros(6,n);
    % Set first column of jacobian to S1
    J(:,1) = S(:,1);
    if n == 1
        return
    end
    A_o = 1;
    % Build jacobian for rest of columns
    for N = 2:n
        A_n = A_o * adjoint(expm(skew4(S(:,N))*theta(N)));
        J(:,N) = A_n * S(:,N);
        A_o = A_n;
    end
end
