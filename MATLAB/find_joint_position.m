function [ p ] = find_joint_position( S, theta, q)

[m,n] = size(S);
p = [];
p = [q(:,1)];
p = [p q(:,2)];

N = eye(4);
for i = 1:n
    N = N*expm(sk6(S(:,i))*theta(i));
    temp = N*[q(:,i+2);1];
    p = [p temp(1:3,1)];
end
end

