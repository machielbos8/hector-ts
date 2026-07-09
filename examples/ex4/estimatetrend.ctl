DataFile              TEST.mom
DataDirectory         ./obs_files
OutputFile            ./mom_files/TEST.mom
periodicsignals       365.25 182.625
estimateoffsets       yes
estimatepostseismic   yes
useRMLE               yes

#--- Power-law noise approximated by GGM with fixed 1-phi
NoiseModels           GGM White
GGM_1mphi             6.9e-06

PhysicalUnit          mm
ScaleFactor           1.0
