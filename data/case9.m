%CASE9    Power flow data for 9 bus, 3 generator case.
%   Please see CASEFORMAT for details on the case file format.
%
%   Based on data from p. 70 of:
%
%   Chow, J. H., editor. Time-Scale Modeling of Dynamic Networks with
%   Applications to Power Systems. Springer-Verlag, 1982.
%   Part of the Lecture Notes in Control and Information Sciences book
%   series (LNCIS, volume 46)
%
%   which in turn appears to come from:
%
%   R.P. Schulz, A.E. Turner and D.N. Ewart, "Long Term Power System
%   Dynamics," EPRI Report 90-7-0, Palo Alto, California, 1974.

%   MATPOWER
%% system MVA base
mpc.baseMVA = 100;

%% bus data
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
	1	3	0	0	0	0	1	1	0	345	1	1.1	0.9;
	2	2	0	0	0	0	1	1	0	345	1	1.1	0.9;
	3	2	0	0	0	0	1	1	0	345	1	1.1	0.9;
	4	1	0	0	0	0	1	1	0	345	1	1.1	0.9;
	5	1	90	30	0	0	1	1	0	345	1	1.1	0.9;
	6	1	0	0	0	0	1	1	0	345	1	1.1	0.9;
	7	1	100	35	0	0	1	1	0	345	1	1.1	0.9;
	8	1	0	0	0	0	1	1	0	345	1	1.1	0.9;
	9	1	125	50	0	0	1	1	0	345	1	1.1	0.9;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf
mpc.gen = [
	1	72.3	27.03	300	-300	1.04	100	1	250	10	0	0	0	0	0	0	0	0	0	0	0;
	2	163	6.54	300	-300	1.025	100	1	300	10	0	0	0	0	0	0	0	0	0	0	0;
	3	85	-10.95	300	-300	1.025	100	1	270	10	0	0	0	0	0	0	0	0	0	0	0;
];

%% branch data
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
	1	4	0	0.0576	0	250	250	250	0	0	1	-360	360;
	4	5	0.017	0.092	0.158	250	250	250	0	0	1	-360	360;
	5	6	0.039	0.17	0.358	150	150	150	0	0	1	-360	360;
	3	6	0	0.0586	0	300	300	300	0	0	1	-360	360;
	6	7	0.0119	0.1008	0.209	150	150	150	0	0	1	-360	360;
	7	8	0.0085	0.072	0.149	250	250	250	0	0	1	-360	360;
	8	2	0	0.0625	0	250	250	250	0	0	1	-360	360;
	8	9	0.032	0.161	0.306	250	250	250	0	0	1	-360	360;
	9	4	0.01	0.085	0.176	250	250	250	0	0	1	-360	360;
];

%% power flow data preparation
nbus = size(mpc.bus,1); % number of buses
nline = size(mpc.branch,1); % number of lines
ngen = size(mpc.gen,1); % number of generators


fbus = mpc.branch(:,1); % from bus of each line
tbus = mpc.branch(:,2); % to bus of each line
rline = mpc.branch(:,3); % resistance(pu) in lines
xline = mpc.branch(:,4); % reactance(pu) in lines
bline = 1i*mpc.branch(:,5); % capacitive susceptance(pu)

zline = rline + 1i*xline;
yline = (1./zline);
ybus = zeros(nbus,nbus);
for k = 1:nline %loop for Ybus
    p = fbus(k);q = tbus(k);
    ybus(p,p)=ybus(p,p)+yline(k)+bline(k)/2;
    ybus(q,q)=ybus(q,q)+yline(k)+bline(k)/2;
    ybus(p,q)=ybus(p,q)-yline(k);
    ybus(q,p)=ybus(p,q);
end
type = mpc.bus(:,2);    % bus type
Vm = mpc.bus(:,8);      % voltage magnitude
Va = mpc.bus(:,9);      % voltage angle
Vbus = mpc.bus(:,8).*exp(1i*mpc.bus(:,9));
Pload = mpc.bus(:,3)/mpc.baseMVA;      % load real power
Qload = mpc.bus(:,4)/mpc.baseMVA;      % load reactive power

Pgen = mpc.gen(:,2)/mpc.baseMVA;  % generator real power
Qgen = mpc.gen(:,3)/mpc.baseMVA;  % generator reactive power

bus_PV = find(type==2);
bus_PQ = find(type==1);

bus_nonslack = [bus_PV;bus_PQ];
nbus_PV = length(bus_PV);
nbus_PQ = length(bus_PQ);
nbus_nonslack = length(bus_nonslack);

B1 = -imag(ybus(bus_nonslack,bus_nonslack));
B11 = -imag(ybus(bus_PQ,bus_PQ));
invB1 = inv(B1);
invB11 = inv(B11);