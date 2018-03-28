function [ t ] = inv_t( m )

% inverse of a transformation matrix
r=[m(1,1) m(1,2) m(1,3);m(2,1) m(2,2) m(2,3);m(3,1) m(3,2) m(3,3)];
p=[m(1,4);m(2,4);m(3,4)];

rn = r.';
pn = -r.'*p;

t = [rn(1,1) rn(1,2) rn(1,3) pn(1);rn(2,1) rn(2,2) rn(2,3) pn(2);rn(3,1) rn(3,2) rn(3,3) pn(3);0 0 0 1];

end
