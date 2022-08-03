function g = g_convolve( stim, kern, frac )
%
% Usage: g = g_convolve( stim, kern, <oversample> )
%
% Efficient form of g_convolution
% Assumes first term in kernel is 0-latency stim 
% and that k is reversed in time
%
% Created by DAB a long time ago

if nargin < 3
  frac = 1;
end

% Make stim into a row
[L1 L] = size(stim);
if L1 ~= 1
  stim = stim';  L = L1;
end

% Upsample stim
stimup = stim(floor((0:L*frac-1)/frac)+1);

g = zeros(1,L*frac);
for i = 1:length(kern)
  %g = g + kern(i)*dist_shift(stimup,i-1,0);
  g = g + kern(i)*stimup;
  stimup = [0 stimup(1:end-1)];
end

% Make into a column
g = g(:);
