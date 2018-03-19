function A = adjoint(T)
% Creates an adjoint matrix from a given transformation matrix
    R = [T(1,1) T(1,2) T(1,3); T(2,1) T(2,2) T(2,3); T(3,1) T(3,2) T(3,3)];
    P = [T(1,4);T(2,4);T(3,4)];
    P_S = skew3(P);
    A = [R [0 0 0; 0 0 0; 0 0 0]; P_S * R R];
end
