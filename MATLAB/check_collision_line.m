function s = check_collision_line(S, p_robot, r_robot, p_obstacle, r_obstacle, theta_a, theta_b, e)
% Calculates whether or not a line from theta a to b will result in a
% collision. Resolution of the check can be adjusted with e. Use e = 0.1
% if unsure.
% Returns 0 if none, a positive integer s in (0,1] if collide
    % number of segments to split the line into
    n = 1 + ceil(norm(theta_a - theta_b)/e);
    s = linspace(0,1,n);
    [~,m] = size(s);
    for i = 1:m
        theta = (1-s(i)) * theta_a + s(i) * theta_b;
        if check_collision(S, p_robot, r_robot, p_obstacle, r_obstacle, theta) == 1
            s = s(i);
            return;
        end
    end
    s = 0;
end
