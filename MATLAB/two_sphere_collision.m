function [ in_collision ] = two_sphere_collision( p1, r1, p2, r2 )
% check if two spheres is in collision given their position and radius
in_collision = (norm(p1-p2) <= r1 + r2);
end
