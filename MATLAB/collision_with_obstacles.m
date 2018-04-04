function [ c ] = collision_with_obstacles( p, r, p_obstacle, r_obstacle)
% Given a position and a radius of one sphere, check if this sphere is in collision with any obstacles in the environment
% return 0 if not in collision, 1 if in collision with any obstacle in the array

[m,n] = size(p_obstacle);
c = 0;

for i = 1:n
    temp = two_sphere_collision(p,r,p_obstacle(:,i),r_obstacle(i));
    if (temp)
        c = 1;
        break
    end
end

end
