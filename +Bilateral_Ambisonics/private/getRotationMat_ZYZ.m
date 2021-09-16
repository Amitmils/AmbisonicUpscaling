function M=getRotationMat_ZYZ(alpha,beta,gamma)
%calculating rotatin matrix
%The rotation axis sequences is Z-Y-Z and the rotation direction is counterclockwise
% with ?, ?, ? as the respective angles. The rotation matrix is M = R1(?)*R2(?)*R3(?)

R1=eye(3);
R1(1:2,1:2)=[cos(alpha) -1*sin(alpha) ;sin(alpha) cos(alpha) ];

R2=eye(3);
R2(1,1) = cos (beta);
R2(1,3) = sin (beta);
R2(3,1) = -1*sin (beta);
R2(3,3) = cos (beta);

R3=eye(3);
R3(1:2,1:2)=[cos(gamma) -1*sin(gamma) ;sin(gamma) cos(gamma) ];

M=R1*R2*R3;