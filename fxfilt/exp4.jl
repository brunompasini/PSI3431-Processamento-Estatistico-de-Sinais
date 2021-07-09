using DSP
using WAV
using PyCall
using PyPlot
using Statistics
using Polynomials
using SampledSignals
using FixedPointNumbers

include("fxfilt.jl");

sig = pyimport("scipy.signal");

fa=40_000;
t=0:(1/fa):3

f0=100*π;
s=0.5*cos.(2*π*f0*t)+0.3*cos.(4*π*300*t)+0.15*cos.(6*π*f0*t)

PotSig=(0.5^2)/2 + (0.3^2)/2 + (0.15^2)/2

B0=5;
σ20=(2.0^(-2*B0))/3

SNRinT = pow2db(PotSig/(σ20));

sq=Fixed{Int16,B0-1}.(s);

ωp=0.07*π
ωr=0.2*π

Ar=41;# foi escolhido 41dB ao inves de 40dB pois ao quantizar os coeficientes no item 4 o filtro saia das especificacoes
δp=0.05;
Ap=-20*log10(1-δp);

N,Wn=sig.ellipord(ωp/π,ωr/π,Ap,Ar);

b,a=sig.ellip(N,Ap,Ar,Wn);

h=PolynomialRatio(b,a)
ω = range(0, π, length = 500)
H=freqz(h,ω)

#item 2
PotRT= σ20*sum(impz(h,500).^2);

SNRsaidaT=pow2db(PotSig/PotRT)

#item 3
y=fxfilt(b,a,s)
yq=fxfilt(b,a,Float64.(sq))

Ein=sq-s;
Eq=yq-y;

PotR=var(Eq);

SNRsaida=pow2db(PotSig/PotR)

#item 4
Bc=12;
σ2=(2.0^(-2*Bc))/3

bq=Fixed{Int16,Bc-1}.(b);
aq=Fixed{Int16,Bc-1}.(a);

hq=PolynomialRatio(Float64.(bq),Float64.(aq));
hdenq = PolynomialRatio([1],Float64.(aq));
Hq=freqz(hq,ω)

ycq=fxfilt(bq,aq,Fixed{Int16,Bc-1}.(s),true)
yqcq=fxfilt(bq,aq,Fixed{Int16,Bc-1}.(sq),true)

#item 5 e 6
PotRcqT= σ20*sum(impz(hq,500).^2)+σ2*sum(impz(hdenq,500).^2);

SNRsaidacqT=pow2db(PotSig/PotRcqT)

Et=yqcq-ycq;

PotRcq=var(Et);

SNRsaidacq=pow2db(PotSig/PotRcq)

#item 7

PotRcanon=(σ20+σ2)*sum(impz(hq,500).^2)+σ2

SNRcanon=pow2db(PotSig/PotRcanon)

# item 8 e 9
h2o = convert(SecondOrderSections,h)
Nsec=length(h2o.biquads)

g=h2o.g

a2o=Vector{Vector{Float64}}(undef,Nsec)
b2o=Vector{Vector{Float64}}(undef,Nsec)
aq2o=Vector{Vector{Fixed{Int64,Bc-1}}}(undef,Nsec)
bq2o=Vector{Vector{Fixed{Int64,Bc-1}}}(undef,Nsec)

Hsosq = Complex.(ones(length(ω)))
for k=1:Nsec
     b2o[k]=coefb(h2o.biquads[k]);
     a2o[k]=coefa(h2o.biquads[k]);
     bq2o[k] = Fixed{Int64,Bc-1}.(b2o[k].*(g.^(1/Nsec)));
     aq2o[k] = Fixed{Int64,Bc-1}.(a2o[k]);
     Hsosq .= Hsosq .* freqz(PolynomialRatio(bq2o[k],aq2o[k]),ω);
end;

sqsos = fxfilt(bq2o[1],aq2o[1],Fixed{Int64,Bc-1}.(sq))
ssos = fxfilt(bq2o[1],aq2o[1],s)

for k=2:Nsec
    global sqsos
    global ssos
    sqsos = fxfilt(bq2o[k],aq2o[k],Fixed{Int64,Bc-1}.(sqsos),true)
    ssos = fxfilt(bq2o[k],aq2o[k],ssos)
end

h1sos=PolynomialRatio(Float64.(bq2o[1]),Float64.(aq2o[1]))
h2sos=PolynomialRatio(Float64.(bq2o[2]),Float64.(aq2o[2]))
h1sosden=PolynomialRatio([1],Float64.(aq2o[1]))
h2sosden=PolynomialRatio([1],Float64.(aq2o[2]))

PotRs1= σ20*sum(impz(h1sos,500).^2)+σ2*sum(impz(h1sosden,500).^2);
PotRsosT=PotRs1*sum(impz(h2sos,500).^2)+σ2*sum(impz(h2sosden,500).^2);

Esos=sqsos-ssos;

PotRsos=var(Esos);

## Plots

#pygui(true)
pygui(false)

# Comparacao sinal original vs quantizado
fig = figure()
subplot(211)
plot(t[100:250], s[100:250])
ylabel("Sinal original")
subplot(212)
plot(t[100:250], sq[100:250])
ylabel("Sinal Quantizado")
xlabel("t")
display(fig)

# Freqz Filtro
fig = figure()
plot(ω/π, 20*log10.(abs.(H)))
plot([0;ωp/π],Ap*[1;1],"r")
plot([ωr/π;1],-40*[1;1],"r")
ylabel("RespFreq H")
xlabel(L"ω/π")
display(fig)

# Comparacao sinais filtrados
fig = figure()
subplot(211)
plot(t[100:250], y[100:250])
title("Saida Filtro Alta Precisao")
ylabel("Sinal Original")
subplot(212)
plot(t[100:250], yq[100:250])
ylabel("Sinal Quantizado")
xlabel(L"t")
display(fig)

# Ruido saida
fig = figure()
plot(t[100:300], Ein[100:300],label="Ruido na Entrada")
title("Filtro de Alta Precisao")
plot(t[100:300], Eq[100:300],label="Ruido na Saida","r")
legend(loc="upper right")
ylabel("Amplitude do Ruido")
xlabel(L"t")
display(fig)

# Freqz Filtro Hq
fig = figure()
plot(ω/π, 20*log10.(abs.(H)),label="Alta Precisao")
plot(ω/π, 20*log10.(abs.(Hq)),label="Quantizado")
plot([0;ωp/π],Ap*[1;1],"r")
plot([ωr/π;1],-40*[1;1],"r")
legend(loc="upper right")
ylabel("RespFreq Hq")
xlabel(L"ω/π")
display(fig)

# Comparacao sinais filtrados Hq
fig = figure()
subplot(211)
plot(t[100:250], ycq[100:250])
title("Saida Filtro Coefs Quantizados")
ylabel("Sinal Original")
subplot(212)
plot(t[100:250], yqcq[100:250])
ylabel("Sinal Quantizado")
xlabel(L"t")
display(fig)

# Ruido saida Hq
fig = figure()
plot(t[100:300], Ein[100:300],label="Ruido na Entrada")
plot(t[100:300], Et[100:300],label="Ruido na Saida","r")
title("Ruidos Coefs Quantizados")
legend(loc="upper right")
ylabel("Amplitude do ruido")
xlabel(L"t")
display(fig)

# Freqz Filtro Seg Ord Sec
fig = figure()
plot(ω/π, 20*log10.(abs.(H)),label="Alta Precs")
plot(ω/π, 20*log10.(abs.(Hsosq)),label="Sec Seg Ord")
plot([0;ωp/π],Ap*[1;1],"r")
plot([ωr/π;1],-40*[1;1],"r")
legend(loc="upper right")
ylabel("RespFreq Hsosq")
xlabel(L"ω/π")
display(fig)

# Comparacao sinais filtrados Seg Ord Sec
fig = figure()
subplot(211)
plot(t[100:250], ssos[100:250])
PyPlot.title("Saida Filtro Secoes Segunda Ordem")
ylabel("Sinal Original")
subplot(212)
plot(t[100:250], sqsos[100:250])
ylabel("Sinal Quantizado")
xlabel(L"t")
display(fig)

#Ruido saida Seg Ord Sec
fig = figure()
plot(t[100:300], Ein[100:300],label="Ruido na Entrada")
plot(t[100:300], Esos[100:300],label="Ruido na Saida","r")
title("Ruidos Seg Ord Secs")
legend(loc="upper right")
ylabel("Amplitude do Ruido")
xlabel(L"t")
display(fig)

# Freqz Filtros
fig = figure()
plot(ω/π, 20*log10.(abs.(H)),label="Alta Precs")
plot(ω/π, 20*log10.(abs.(Hq)),label="Quant")
plot(ω/π, 20*log10.(abs.(Hsosq)),label="Seg Ord Sec")
plot([0;ωp/π],Ap*[1;1],"r")
plot([ωr/π;1],-40*[1;1],"r")
ylabel("RespFreq Hsosq")
xlabel(L"ω/π")
display(fig)
