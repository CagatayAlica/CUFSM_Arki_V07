import math

minimas = []

fy = 50.76  # ksi 280 MPa

sf = 0.290  # in3
ratcl = 0.759
ratdb = 0.711

Mne = sf * fy
Mcrl = Mne * ratcl
Mcrd = Mne * ratdb

lambdal = math.sqrt(Mne / Mcrl)

if lambdal <= 0.776:
    Mnl = Mne
else:
    Mnl = (1 - 0.15 * math.pow(Mcrl / Mne, 0.4)) * math.pow(Mcrl / Mne, 0.4) * Mne

lambdad = math.sqrt(Mne / Mcrd)

if lambdal <= 0.673:
    Mnd = Mne
else:
    Mnd = (1 - 0.22 * math.pow(Mcrd / Mne, 0.5)) * math.pow(Mcrd / Mne, 0.5) * Mne

print(f"Mne = {Mne}\nMnl = {Mnl}\nMnd = {Mnd}")
print(f"Mne = {Mne / 1.67 * 0.112985}\nMnl = {Mnl / 1.67 * 0.112985}\nMnd = {Mnd / 1.67 * 0.112985}")