function [out]= Mitigation_tetra 

data=[9900	5400	1350	1	1500
6600	8100	315	1	700
0	360	1440	1	500
0	3240	864	1	200
0	1620	1188	1	1100
0	7200	2520	1	700
0	0	200	0	0
35200	23040	2592	0	3600
77440	57600	5184	0	7200
10000	2880	72	0	13000];
data=data';

n_var=10;
n_c=2^n_var-1;
k=flip(decimalToBinaryVector([0:1:n_c]),2);
% k=k(6,:);
% k1=zeros(1,n_var);
% k2=ones(1,n_var);
% k3=k;
% k=[k1;k2;k3]; 
c=size(k,1); 

capex=k.*data(1,:);
capex=capex/sum(capex(c,:));
capex=(1-sum(capex,2))*100;

opex=k.*data(2,:);
opex=opex/sum(opex(c,:));
opex=(1-sum(opex,2))*100;

redo=k.*data(3,:);
redo=redo/sum(redo(c,:));
redo=(1-sum(redo,2))*100;

penalty=k.*data(4,:);
penalty=penalty/sum(penalty(c,:));
penalty=(1-sum(penalty,2))*100;

vvu=k.*data(5,:);
vvu=vvu/sum(vvu(c,:));
vvu=(1-sum(vvu,2))*100;

w=[2 2 3 1 1]./10;

P=[capex opex redo penalty vvu];

out=-Tetra_solver2(w,P);
assignin('base','out',out)

end