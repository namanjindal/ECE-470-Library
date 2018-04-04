function [ c ] = collision_with_obstacles( p, r_robot, p_obstacle, r_obstacle)

[m,n] = size(p_obstacle);
c = 0;

for i = 1:n
    temp = two_sphere_collision(p,r_robot,p_obstacle(:,i),r_obstacle(i));
    if (temp)
        c = 1;
        break
    end
end

end

