function closestNode = find_closest_node(node, tree)
% Finds the node in a tree that's has the smallest theta difference norm to
% a given node. "node" is expected as a nx1 matrix of joint angles, where n is the number
% of robot joints. "tree" is expected as a nxm matrix [node1 node2 ... nodeM], where m is the
% number of nodes in the tree.
    [~,m] = size(tree);
    closestNode = tree(:,1);
    for i = 2:m
        if norm(node - closestNode) > norm(node - tree(:,i)) && isequal(node, tree(:,i)) == 0
            closestNode = tree(:,i);
        end
    end
end
