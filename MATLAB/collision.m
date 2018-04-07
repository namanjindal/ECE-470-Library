% Given the screw axis, initial pose, centers and radius of all spheres and
% initial and final joint angles, returns 1 if there is a collision and 0
% if there is not a collision

function col=collision(S,M,p_robot,r_robot,p_obstacle,r_obstacle,theta_start,theta_goal)

    [n_rows,n_columns]=size(S);
    col=0;
    
    for m=1:1000
        t=(1-m/1000)*theta_start+m/1000*theta_goal;
        
        for p=1:n_columns
            B(:,:,p)=expm(skew2(S(:,p))*t(p));
        end
        
        P=[p_robot; ones(1,n_columns+2)];
        
        for p=3:n_columns+2
            Bprod=eye(4);
            for a=1:(p-2)
                Bprod=Bprod*B(:,:,a);
            end
            P(:,p)=Bprod*P(:,p);
        end
        
        P=P(1:3,:);
        
        for j=1:(n_columns+2)
            for k=(j+1):(n_columns+2)
                if ((norm(P(:,j)-P(:,k))<=r_robot(j)+r_robot(k)))
                    col=1;
                    break;
                end
            end
            for t=1:size(r_obstacle,2)
                if (norm(P(:,j)-p_obstacle(:,t))<=(r_robot(j)+r_obstacle(t)))
                    col=1;
                    break;
                end
            end
            if(col==1)
                break;
            end
        end
        if(col==1)
            break;
        end
    end
end