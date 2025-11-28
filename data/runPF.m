clc
clear all
run("case9.m")
%% random power consumption of loads 
rrand = (0.5 + (1.5 - 0.5) * rand(2*nbus,1));
Pload_random = Pload.*rrand(1:nbus);
Qload_random = Qload.*rrand(nbus+(1:nbus));
%% power flow algorithm 
% https://www.mathworks.com/matlabcentral/fileexchange/119333-fast-decoupled-load-flow-matlab-code
Psp = -Pload_random; Psp(mpc.gen(:,1)) = Psp(mpc.gen(:,1))+Pgen; 
Qsp = -Qload_random; Qsp(mpc.gen(:,1)) = Qsp(mpc.gen(:,1))+Qgen; 
Ssp = Psp +1i*Qsp;

dP =zeros(nbus_nonslack,1); dQ = zeros(nbus_PQ,1);
Va =zeros(nbus,1); Vm = ones(nbus,1);
MV = [dP; dQ];
iter = 1;
tol = 1;
while (tol>1e-8)
    S = Vbus.*conj(ybus*Vbus);
    S1 = S-Ssp;
    dP = real(S1(bus_nonslack));
    dQ = imag(S1(bus_PQ));
    
    dVa = -invB1*dP;  
    dVm = -invB11*dQ; 
    MV = [dP; dQ];
    Va(bus_nonslack) = Va(bus_nonslack)+ dVa;
    Vm(bus_PQ) = Vm(bus_PQ)+ dVm;
    Vbus = Vm.*exp(1i*Va);
    iter = iter+1;
    tol = max(abs(MV));
end
%% Neural network input: random power consumption of loads 
NN_input = [Pload_random(bus_PQ); Qload_random(bus_PQ)];
%% Neural network output: power flow solution
NN_output = [Va;Vm];