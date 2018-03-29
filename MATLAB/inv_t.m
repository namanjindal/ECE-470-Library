function [t] = inv_t(m)
% Finds the inverse of a transformation matrix
% Same as inv(), but faster
    r = m(1:3,1:3);
    p = m(1:3,4);

    rn = r.';
    pn = -r.'*p;

    t = [rn pn; 0 0 0 1];
end
