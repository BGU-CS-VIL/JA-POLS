
% -------------------------------------------------------------------------------------------

disp('Running matlab code...');

clear();

height = 250;
width = 420;

prpca_path = strcat(pwd, '/2_learning/BG/PRPCA/');
addpath(strcat(pwd, '/2_learning/BG/PRPCA/src/'));

% Load x_zeromean data:
load x.mat x
load w.mat w

% img = reshape(x(:, 1),[height, width, 3]);
% figure();
% imshow(img,[]);
% title('x');
% 
% img = reshape(w(:, 1),[height, width, 3]);
% figure();
% imshow(img,[]);
% title('w');

%Ytil = load(strcat(prpca_path, 'x.mat'));

%y = reshape(x, [size(x,1)*size(x,2)*size(x,3), size(x,4)]);

% Run robustPCA.m on it:

lambdaS = 1 / sqrt(max(size(x)));   % y size: dxN

m = any(w,2);
Ytil = x(m,:);
Mtil = w(m,:);

opts = struct();
opts.M = Mtil;
opts.nIters = 25;

[Ltil, Stil] = robustPCA(Ytil,1,lambdaS,opts);

% Embed components back into full space
Lhat = zeros(size(x));
Shat = zeros(size(x));
Lhat(m,:) = Ltil;
Shat(m,:) = Stil;

Lreg = reshape(Lhat,[height, width, 3, size(x,2)]);
Sreg = reshape(Shat,[height, width, 3, size(x,2)]);
M = logical(reshape(w,[height, width, 3, size(x,2)]));

% img = Lreg(:, :, :, 1);
% figure();
% imshow(img,[]);
% title('L - before improvement');

[Lreg, Sreg] = adjustLS(Lreg,Sreg,M);

% img = Lreg(:, :, :, 1);
% figure();
% imshow(img,[]);
% title('L - after improvement');


save(strcat(prpca_path, 'L.mat'), 'Lreg');  % Lreg size: hxwx3xN

disp('Finish matlab code.');

% -------------------------------------------------------------------------------------------

