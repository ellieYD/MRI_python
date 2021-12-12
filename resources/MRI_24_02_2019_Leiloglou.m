
%{
Reading Week Practical  MRI/Matlab Focus
Non-Ionising Imaging
Dr. Neal Bangerter, February 2019

Maria Leiloglou 24/02/2019 solution 
%}
% PART 1
[x, y] = meshgrid(-4:0.05:4);  %type ?help meshgrid? if you haven?t seen this before
  s1 = exp(-pi*(x.^2 + y.^2));
  imshow(s1, []);
  
  [x, y] = meshgrid(-4:0.05:4 + 10*eps);  %type ?help eps? (a useful thing to know)
  s2 = sin(-pi*x).*sin(-pi*y)./(pi^2*x.*y);
  imshow(s2, []);

  kx_0 = 0.25;
  ky_0 = 0.5;
  s3 = exp(j*2*pi*(kx_0*x + ky_0*y));
  figure(1);
  subplot(1,2,1);    %A useful thing to know for putting multiple graphs in a single figure
  imshow(real(s3), []);   %s3 is complex, so let?s look at both the real part?
  subplot(1,2,2);
  imshow(imag(s3), []);  %.?and the imaginary part.
% (e)	Try several different values for kx_0 and ky_0.  What do these parameters represent?
kx_0 = 0.25;
ky_0 = 0.5;
s3 = exp(j*2*pi*(kx_0*x + ky_0*y));%meshgrid always the same/when kx small s3 depends less on x, when large s3 changes faster with x 
figure(2);subplot(221);imshow(real(s3), []);
kx_0 = 0.025;
ky_0 = 0.05;
s3 = exp(j*2*pi*(kx_0*x + ky_0*y));
subplot(222);imshow(real(s3), []);
kx_0 = 2.5;
ky_0 = 5;
s3 = exp(j*2*pi*(kx_0*x + ky_0*y));
subplot(223);imshow(real(s3), []);
>> kx_0 = 25;
ky_0 = 50;
s3 = exp(j*2*pi*(kx_0*x + ky_0*y));
subplot(224);imshow(real(s3), []);


%How about creating rect(x,y)?  Try the following code:

  s4 = double(abs(x)<0.5 & abs(y)<0.5);
  figure(3);
  imshow(s4,[]);

  
  %{
  Remember that x and y are 2D matrices returned by the meshgrid function.  
The expression abs(x)<0.5 returns a 2D matrix whose values are either 1 or
 0 depending on whether the corresponding element in x  has an absolute 
value less than 0.5.  The  & performs a logical AND operation.  
Puzzle through this expression and figure out why it works if it isn?t clear.  
Why do I cast it to a double afterwards?
  %}
  %%
  %Part 2:  Using the 2D Fast Fourier Transform in Matlab
  
    S1 = fftshift(fft2(s1)); % fft2 computes the 2D Fourier Transform.  
    %The fftshift command is needed to move the origin of spatial frequency
    %space to the center of the image.
    S3 = fftshift(fft2(s3));
    S4 = fftshift(fft2(s4));
figure(4);subplot(321);imshow(s1,[]);
subplot(322);imshow(abs(S1),[]);
subplot(323);imshow(s3,[]);
subplot(324);imshow(abs(S3),[]);
subplot(325);imshow(s4,[]);
subplot(326);imshow(abs(S4),[]);

f1=imread('head.jpg');
f1 = double(f1);
figure(5);imshow(f1,[]);
%Compute the 2D Fourier Transform F2 of the image, and display the 
%logarithm of its absolute value and its phase.

F1 = fftshift(fft2(f1));
  figure;
  subplot(2,1,1);
  imshow(log(abs(F1)),[]);
  subplot(2,1,2);
  imshow(angle(F1),[]); %?angle? in Matlab computes phase of a complex number
%{
Unlike some of the 2D signals we examined before, most real
images (like MRI images) have the vast majority of the energy concentrated 
in the very low spatial frequencies
(the center of the k-space image). If you don?t take the logarithm, you?ll 
often just see a bright spot in the
middle of your 2D Fourier transform, and won?t see any detail in the higher 
spatial frequencies.
  
  
  The bright
spot in the center corresponds to the ?DC component? of the image, which is
 average intensity across all of
the pixels in the image.
%}
  
%Using subplot, create a plot showing (1) the original image f1, (2) the 
%log of the magnitude of F1 (the 2D
%Fourier transform of f1), and (3) the phase of F1. You can add titles to 
%each graph with the ?title? function.
  
  figure(6);subplot(311);imshow(f1,[]);title('head mri image');
  subplot(312);imshow(log(abs(F1)),[]);title(' Magnitude of FT head mri image');
  subplot(313);imshow(angle(F1),[]);title('Angle of FT head mri image');
  
%low-pass filtering the image by multiplying F2 by a rect function.  

  rect1 = zeros(256);
  rect1(112:144,112:144) = 1;
  figure(7);subplot(311);
  imshow(rect1,[]);hold on; title('the low pass filter');	%Display your low-pass rect function to make sure you?ve done it correctly
  F1_lowpass = F1.*rect1; %keep only centre of k space thus only colour/contrast/ low spatial frequency of the image
  subplot(312);imshow(log(abs(F1_lowpass)),[]);title('low-pass filtered k-space');%Display your k-space data after applying the low-pass filter
  f1_lowpass = ifft2(fftshift(F1_lowpass));  % Do the inverse 2DFT
  subplot(313);imshow(abs(f1_lowpass),[]);title('low-pass filtered head image');%And display the low-pass filtered image
imwrite(f1_lowpass,'low pass head.jpg');

%Now try high-pass filtering the image. (Hint: try multiplying F2 by 1 - rect1.)

rect2=ones(256);
rect2(112:144,112:144) = 0;
 figure(8);subplot(311);
  imshow(rect2,[]);hold on; title('the high pass filter');
  F2_highpass = F1.*rect2; %keep only peripheries of k space thus only details/ high spatial frequency of the image
  subplot(312);imshow(log(abs(F2_highpass)),[]);title('high-pass filtered k-space');%Display your k-space data after applying the low-pass filter
  f2_highpass = ifft2(fftshift(F2_highpass));  % Do the inverse 2DFT
  subplot(313);imshow(abs(f2_highpass),[]);title('high-pass filtered head image');%And display the high-pass filtered image
imwrite(f2_highpass,'high pass head.jpg');

%{ 

d

In the following two sections, we?re going to demonstrate 
some of the properties of the Fourier transform
using the 2D rect signal s4 that you generated previously.

Translate (or shift) your rect image s4 using the ?circshift? command in Matlab. Shift it 40 pixels in the x
direction and 20 pixels in the y direction. Now compute the 2D Fourier transform of the shifted image,
and compare its magnitude and phase to the magnitude and phase of the Fourier transform of the
original image.

%}
%original s4
s4 = double(abs(x)<0.5 & abs(y)<0.5);
figure(9);subplot(3,2,1);imshow(s4,[]);title('original s4');
 S4 = fftshift(fft2(s4));%FT of the original s4
subplot(3,2,3); imshow(log(abs(S4)),[]);title('Magnitude of FT original s4'); 
 subplot(3,2,5); imshow(angle(S4),[]);title('Phase of FT original s4'); 

 %shifted s4
s5=circshift(s4,40,1);s6=circshift(s5,20,2);
subplot(3,2,2);imshow(s6,[]);title('shifted s4');
S6 = fftshift(fft2(s6));%FT of the shifted original s4
subplot(3,2,4); imshow(log(abs(S6)),[]);title('Magnitude of FT shifted4'); 
subplot(3,2,6);imshow(angle(S6),[]);title('angle of FT shifted4');

%shift in space results in linear modulation of phase in k-space.


%{

e

Now try rotating your original 2D rect s4 using the ?imrotate? command in 
Matlab. Compare the Fourier
transforms of the original image and the rotated image, and 
explain your results.
%}
s4 = double(abs(x)<0.5 & abs(y)<0.5);
figure(10);subplot(3,2,1);imshow(s4,[]);title('original s4');
 S4 = fftshift(fft2(s4));%FT of the original s4
subplot(3,2,3); imshow(log(abs(S4)),[]);title('Magnitude of FT original s4'); 
 subplot(3,2,5); imshow(angle(S4),[]);title('Phase of FT original s4'); 
 
 %rotated s4
 s7=imrotate(s4,10);
subplot(3,2,2);imshow(s7,[]);title('10 degrees rotated s4');
S7 = fftshift(fft2(s7));%FT of the rotated original s4
subplot(3,2,4); imshow(log(abs(S7)),[]);title('Magnitude of FT rotated4'); 
subplot(3,2,6);imshow(angle(S7),[]);title('angle of FT rotated4');
%imrotate increases the size of the image and causes the same 10 degree
%rotation in the fourier domain.


%{
f
Finally, take the Fourier transform of your shifted 2D rect and zero out
almost half of your data. For example, if the Fourier transform of your 
shifted rect is S6, you
could type: S6(1:80,:) = 0

%}
[x, y] = meshgrid(-4:0.05:4 + 10*eps); 
s4 = double(abs(x)<0.5 & abs(y)<0.5);
s5=circshift(s4,40,1);s6=circshift(s5,20,2);
S6 = fftshift(fft2(s6));
S6(1:80,:) = 0; %zeroing half of the k space
figure(11);
subplot(511);imshow(log(abs(S6)),[]); title('magnitude half k-space');
subplot(512);imshow(angle(S6),[]); title('phase half k-space');
 s6half =(ifft2(S6));

 subplot(513);imshow(abs(s6half),[]); title('s4 from half k space');%why complex?
 
 %Can you devise a
%scheme (maybe using conjugate symmetry) to restore the zeroed-out values
%in the Fourier domain? Try it and see if you can get something that looks 
%more like your original shifted rectangle back.

S6(1:80,:)=conj(S6(161:-1:82,161:-1:1));%use conjugate symentry
subplot(514);imshow(log(abs(S6)),[]); title('magnitude restored k-space');
s6restored=(ifft2(S6));

subplot(515);imshow(abs(s6restored),[]); title('s4 from restored k-space');
%%
%{
 PART 3 Effect of Point Spread Function (PSF) on Image Quality
a
%}

f1 = imread('shepp256.png');
f1 = double(f1);F1=fftshift(fft2(f1));

h1=imread('h1.png');
h1=double(h1);H1=fftshift(fft2(h1));

h2=imread('h2.png');
h2=double(h2);H2=fftshift((fft2(h2)));

h3=imread('h3.png');
h3=double(h3);H3=fftshift((fft2(h3)));



figure(12);
FTpsf1=F1.*H1;psf1=ifftshift(ifft2(FTpsf1));
subplot(221);imshow(abs(psf1),[]);title('convolved with first PSF');
FTpsf2=F1.*H2;psf2=ifftshift(ifft2(FTpsf2));
subplot(222);imshow(abs(psf2),[]);title('convolved with second PSF');
FTpsf3=F1.*H3;psf3=ifftshift(ifft2(FTpsf3));
subplot(223);imshow(abs(psf3),[]);title('convolved with third PSF');
H5=H1.*H2.*H3;
h5=(ifft2(ifftshift(H5)));


FTpsf5=F1.*H5;psf5=ifftshift(ifft2(FTpsf5));
subplot(224);imshow(abs(psf5),[]);title('convolved with all PSFs');

figure(13);
subplot(221);imshow(h1,[]);title('first PSF');

subplot(222);imshow(h2,[]);title('second PSF');

subplot(223);imshow(h3,[]);title('third PSF');


subplot(224);imshow(abs(h5),[]);title('all PSFs');

%c
H4=imread('H4.png');
H4=double(H4);h4=ifftshift(ifft2(H4));

FTpsf4=F1.*H4;psf4=(ifft2(FTpsf4));
figure(14);
subplot(221);imshow(log(abs(H4)),[]);title('magnitude of FD of exponential decay');
subplot(222);imshow(abs(h4),[]);title('SD of exponential decay');
subplot(223);imshow(abs(psf4),[]);title('convolved with exponential decay');
subplot(224);imshow(f1,[]);title('initial phantom image');
%%
%{
PART 4
Measuring Signal-to-Noise Ratio (SNR) and Contrast-to-Noise Ratio (CNR) from Images
%}
clc;
clear all;
close all;
f1 = imread('shepp256.png');
f1 = double(f1);F1=fftshift(fft2(f1));
n1 = poissrnd(80, 256, 256);
N1=fftshift(fft2(n1));

m=mean(n1(:));
v=var(n1(:));

F1poisson=F1 + N1;
f1poisson=ifft2(ifftshift(F1poisson));
im=abs(f1poisson);
%im=uint8(im);
figure(15);
imshow(im,[]);
imwrite(abs(f1poisson),'phantompoisson.jpg');

%choose pixels

%FOR TISSUE 1

windo=15;
signal=zeros(windo,windo);
background=zeros(windo,windo);
waitfor(msgbox('Choose a Tissue 1 Pixel'));
figure(16);
set(gcf,'Units','normalized','outerposition',[0 0 1 1]);
[c1,r1,~]=impixel(im,[]);
close figure 16
waitfor(msgbox('Choose a Background Pixel'));
figure(17);
set(gcf,'Units','normalized','outerposition',[0 0 1 1]);
[c2,r2,~]=impixel(im,[]);
close figure 17

signal=im(r1:r1+windo-1,c1:c1+windo-1);
background=im(r2:r2+windo-1,c2:c2+windo-1);
signal= double(signal);
background= double(background);
signal=reshape(signal,1,numel(signal));
background=reshape(background,1,numel(background));
signalmean=mean(signal);
backgroundmean=mean(background);
signalsd=std(signal);
backgroundsd=std(background);
SNR1=(signalmean/backgroundsd);

%for tissue 2
windo=15;
signal=zeros(windo,windo);
background=zeros(windo,windo);
waitfor(msgbox('Choose a Tissue 2 Pixel'));
figure(18);
set(gcf,'Units','normalized','outerposition',[0 0 1 1]);
[c1,r1,~]=impixel(im,[]);
close figure 18
waitfor(msgbox('Choose a Background Pixel'));
figure(19);
set(gcf,'Units','normalized','outerposition',[0 0 1 1]);
[c2,r2,~]=impixel(im,[]);
close figure 19

signal1=im(r1:r1+windo-1,c1:c1+windo-1);
background=im(r2:r2+windo-1,c2:c2+windo-1);
signal1= double(signal1);
background= double(background);
signal1=reshape(signal1,1,numel(signal1));
background=reshape(background,1,numel(background));
signal1mean=mean(signal1);
backgroundmean=mean(background);
signal1sd=std(signal1);
backgroundsd=std(background);
SNR2=(signal1mean/backgroundsd);

%find CNR

CNR=(signalmean-signal1mean)/(backgroundsd);
%%

%PART 5
clc;
clear all;
close all;
f1 = imread('shepp256.png');
f1 = double(f1);
F1=fftshift(fft2(f1));
n3 =normrnd(0, 5000, 256, 256) + 1i*normrnd(0, 5000, 256, 256);
N3=fftshift(fft2(n3));


F1BIVARIATE=F1 + n3;
f1bivariate=ifft2(ifftshift(F1BIVARIATE));
im2=abs(f1bivariate);
%im=uint8(im);
figure(20);
imshow(abs(f1bivariate),[]);
imwrite(abs(f1bivariate),'phantombivariate.jpg');


%choose pixels

%FOR TISSUE 1

windo=15;
signal=zeros(windo,windo);
background=zeros(windo,windo);
waitfor(msgbox('Choose a Tissue 1 Pixel'));
figure(21);
set(gcf,'Units','normalized','outerposition',[0 0 1 1]);
[c1,r1,~]=impixel(im2,[]);
close figure 21
waitfor(msgbox('Choose a Background Pixel'));
figure(22);
set(gcf,'Units','normalized','outerposition',[0 0 1 1]);
[c2,r2,~]=impixel(im2,[]);
close figure 22

signal=im2(r1:r1+windo-1,c1:c1+windo-1);
background=im2(r2:r2+windo-1,c2:c2+windo-1);
signal= double(signal);
background= double(background);
signal=reshape(signal,1,numel(signal));
background=reshape(background,1,numel(background));
signalmean=mean(signal);
backgroundmean=mean(background);
signalsd=std(signal);
backgroundsd=std(background);
SNR1=(signalmean/backgroundsd);

%for tissue 2
windo=15;
signal=zeros(windo,windo);
background=zeros(windo,windo);
waitfor(msgbox('Choose a Tissue 2 Pixel'));
figure(23);
set(gcf,'Units','normalized','outerposition',[0 0 1 1]);
[c1,r1,~]=impixel(im2,[]);
close figure 23
waitfor(msgbox('Choose a Background Pixel'));
figure(24);
set(gcf,'Units','normalized','outerposition',[0 0 1 1]);
[c2,r2,~]=impixel(im2,[]);
close figure 24

signal1=im2(r1:r1+windo-1,c1:c1+windo-1);
background=im2(r2:r2+windo-1,c2:c2+windo-1);
signal1= double(signal1);
background= double(background);
signal1=reshape(signal1,1,numel(signal1));
background=reshape(background,1,numel(background));
signal1mean=mean(signal1);
backgroundmean=mean(background);
signal1sd=std(signal1);
backgroundsd=std(background);
SNR2=(signal1mean/backgroundsd);

%find CNR

CNR=(signalmean-signal1mean)/(backgroundsd);
