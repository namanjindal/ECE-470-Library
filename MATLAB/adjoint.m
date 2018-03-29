function A = adjoint(T)
% Creates an adjoint matrix from a given transformation matrix
    R = T(1:3,1:3);
    P = T(1:3,4);
    P_S = skew3(P);
    A = [R [0 0 0; 0 0 0; 0 0 0]; P_S * R R];
end
