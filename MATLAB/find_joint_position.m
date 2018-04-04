function [ p ] = find_joint_position( S, theta, q)
% Find the positions of each joint given S, theta and initial position q
[m,n] = size(S);
p = [];
p = [q(:,1)];
p = [p q(:,2)];

N = eye(4);
for i = 1:n
    N = N*expm(skew4(S(:,i))*theta(i));
    temp = N*[q(:,i+2);1];
    p = [p temp(1:3,1)];
end
end
