% clc;
% close all
% clear all
clearvars
addpath(genpath('./l1_ls_matlab'));
addpath(genpath('./GPSR_6.0'));


%% Parameters
n = 32; %PIXELSIZE, up to 96 can be computational fine, from n>64 it takes some time
nn = n*n;
nSpeckles = round(nn*0.075); % number of speckle patterns used for illumination

X = (1:n);
Y = (1:n);

%% 2D Sample
pxl = 120e-9; %real pixel size
Sample1 = double(imread('test2.png'));
% Sample1 = rgb2gray(Sample1);


%% if n < than the image size (Sample1 = 256x256), than just take an n x n size area in the center of the image
sizeSam = size(Sample1,1);
if n < sizeSam
    i0 = sizeSam/2;
    j0 = sizeSam/2;
    Sample = Sample1((i0-n/2+1):(i0+n/2),(j0-n/2+1):(j0+n/2));
else
    Sample = Sample1;
end

amountOfInfo = find(Sample ~= 0);
numOfInfo = size(amountOfInfo,1);
sparsity = numOfInfo * 100 / nn;


%% Gauss: generate a Gaussian beam for point-by-point scanning. For comparison
lambda = 0.532e-6;
NA = 0.22;
FWHM_gauss = lambda/2/NA/pxl;
sigma = FWHM_gauss/(2*sqrt(2*log(2)));
X0 = n/2;
Y0 = n/2;

[XX, YY] = meshgrid(X,Y);
Gauss_focus0 = exp(-(((XX - X0)).^2 + (YY - Y0).^2)/(2*sigma^2));

Igauss_fft = abs(fftshift(fft2(Gauss_focus0)));
cutoff_R = sum(sum(Igauss_fft(end/2,:) >= 2e-1)/4);

Gauss_scan = zeros(n,n);

for ind_y = 1:n
    for ind_x = 1: n
        Gauss_focus = exp(-(((XX - X(ind_x))).^2 + (YY - Y(ind_y)).^2)/(2*sigma^2));
        Gauss_scan(ind_y,ind_x) = sum(sum(Sample .* Gauss_focus));
    end
end


%% Illumination pattern. Generate 2D nSpeckles random speckle patterns
I_speckle_mtrx = speckleSIM(n, nSpeckles, cutoff_R, 2); % 1 for 1D speckles

h = waitbar(0,'Please wait...creating patterns, recon. matrix, etc...');
for kk = 1 : nSpeckles
   A(kk,:) = reshape(I_speckle_mtrx(:,:,kk),nn,1); % reshape the speckle patterns to 1D (and add to the 'sensing' matrix 'A')
   b(kk,1) = sum(sum(Sample .* I_speckle_mtrx(:,:,kk))); % multiply each speckle pattern with the sample and record the resulting intensity/observations 'b'
   waitbar(kk/nSpeckles,h)
end
close(h)

comFac = nSpeckles * 100/ nn;
measRatio = nSpeckles / numOfInfo;

%% BASIS TRANSFORM
%filename = ['B_64_Kronecker.csv'];
%filename = ['B_64_regular.csv'];
%filename = ['B_CIFAR_kron.csv'];
filename = ['B_CIFAR_regular.csv'];
B = csvread(filename);

x0 = A'*b;
%% GPSR REGULAR
tic
disp('GPSR Basic starts...')
    tau = 0.1; % a non-negative real parameter
    Ip_gpsr = GPSR_Basic(b, A, tau);
toc
%% GPSR BASIS TRANSFORM
A = A*B;

tic
disp('GPSR Basic starts...')
    tau = 0.1; % a non-negative real parameter
    Ip_gpsrB = B*GPSR_Basic(b, A, tau);
toc
%% create (or reshape to) images
Ip_initGuess0 = reshape(x0,n,n);
Ip_gpsr = reshape(Ip_gpsr,n,n);
Ip_gpsrB = reshape(Ip_gpsrB,n,n);

% calculate correlation between reconstructed image and original sample
cor_Ip_gpsr = corr2(Sample, Ip_gpsr);
cor_Ip_gpsrB = corr2(Sample, Ip_gpsrB);
% calculate similarity between reconstructed image and original sample
ssim_Ip_gpsr = ssim(Ip_gpsr, Sample);
ssim_Ip_gpsrB = ssim(Ip_gpsrB, Sample);
%% output figure
image_fft = abs(fftshift(fft2(Ip_gpsr)))
image_fftB = abs(fftshift(fft2(Ip_gpsrB)))
image_fft_error = abs(fftshift(fft2(Sample))-fftshift(fft2(Ip_gpsr)))
image_fftB_error = abs(fftshift(fft2(Sample))-fftshift(fft2(Ip_gpsrB)))
% Maxs/mins to equal colorbars:
mini = @(X,Y) min(min(X,[],"all"),min(Y,[],"all"));
maxi = @(X,Y) max(max(X,[],"all"),max(Y,[],"all"));

h = figure
set(gcf, 'WindowState', 'maximized');
subplot (3,2,1), imagesc(Sample), axis image ,colorbar, title('sample')
subplot (3,2,2), imagesc(Ip_initGuess0), axis image, colorbar, title(['init guess'])
subplot (3,2,3), imagesc(Ip_gpsrB), axis image, colorbar, clim([mini(Ip_gpsrB,Ip_gpsr) maxi(Ip_gpsrB,Ip_gpsr)]), title(['GPSR Basis transform, corr: ', num2str(cor_Ip_gpsrB),',ssim: ', num2str(ssim_Ip_gpsrB)])
subplot (3,2,4), imagesc(Ip_gpsr), axis image, colorbar, clim([mini(Ip_gpsrB,Ip_gpsr) maxi(Ip_gpsrB,Ip_gpsr)]), title(['GPSR Basic, corr: ', num2str(cor_Ip_gpsr),',ssim: ', num2str(ssim_Ip_gpsr)])
%subplot (4,2,5), imagesc(image_fftB), axis image, colorbar, clim([mini(image_fftB,image_fft) maxi(image_fftB,image_fft)]), title('FFT B-transform GPSR ')
%subplot (4,2,6), imagesc(image_fft), axis image, colorbar, clim([mini(image_fftB,image_fft) maxi(image_fftB,image_fft)]), title('FFT GPSR')
subplot (3,2,5), imagesc(image_fftB_error), axis image, colorbar, clim([mini(image_fftB_error,image_fft_error) maxi(image_fftB_error,image_fft_error)]), title('FFT B-transform error', sum(image_fftB_error, "all"))
subplot (3,2,6), imagesc(image_fft_error), axis image, colorbar, clim([mini(image_fftB_error,image_fft_error) maxi(image_fftB_error,image_fft_error)]), title('FFT GPSR error', sum(image_fft_error, "all"))
sgtitle(['\lambda = ' , num2str(lambda), ', NA ', num2str(NA),', pxlsize: ', num2str(pxl), ', comp: ', num2str(comFac),'%, contRatio: ', num2str(sparsity),'%, numPat2Info: ', num2str(measRatio),', numOfPxl: ', num2str(numOfInfo)])
