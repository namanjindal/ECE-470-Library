function isCollide = check_collision(S, p_robot, r_robot, p_obstacle, r_obstacle, theta)
% Checks whether or not the new position of n joints when moved by a matrix of nx1 thetas
% will collide with itself or the environment.
% Returns 1 if any collision is detected, 0 if none.
    % Updating positions of spheres on the robot using S and theta
    [m,~] = size(theta);
    for i = 1:m
        p = find_fk(S(:,1:i), theta(1:i,:), [p_robot(:,i+2);1]);
        p_robot(:,i+2) = p(1:3,:);
    end
    % Checking for collisions between the robot and itself
    [~,n] = size(p_robot);
    for i = 1:n
        for j = i+1:n
            if norm(p_robot(:,i) - p_robot(:,j)) <= r_robot(i) + r_robot(j)
                isCollide = 1;
                return;
            end
        end
    end
    % Checking for collisions between the robot and external objects
    [~,k] = size(p_obstacle);
    for i = 1:k
        for j = 1:n
            if norm(p_obstacle(:,i) - p_robot(:,j)) <= r_obstacle(i) + r_robot(j)
                isCollide = 1;
                return
            end
        end
    end
    isCollide = 0;
end
