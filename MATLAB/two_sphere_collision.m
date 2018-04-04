function [ in_collision ] = two_sphere_collision( p1, r1, p2, r2 )
in_collision = (norm(p1-p2) <= r1 + r2);
end

