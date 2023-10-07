function des2dke(inDir)
% DES2DKE  Convert designer output to DKE inputs
%   DES2DKE('input directory') creates a folder called 'DKE' within
%   Designer's output folder and prepares all files for DKE. It modifies
%   DWIs, BVALs and BVECs to ensure compatibility with DKE.
%
%   All B0s in DWIs in removed and Designer's B0 is inserted into the first
%   index. This change is also applied to BVAL and BVEC so they reflect
%   changes made to DWI.
%
%   Author: Siddhartha Dhiman
%   Email:  dhiman@musc.edu

%% Load Paths
b0_Path = fullfile(inDir,'B0.nii');
dwi_Path = fullfile(inDir,'dwi_preprocessed.nii');
bval_Path = fullfile(inDir,'dwi_preprocessed.bval');
bvec_Path = fullfile(inDir,'dwi_preprocessed.bvec');
mask_Path = fullfile(inDir,'brain_mask.nii');
dke_Path = fullfile(inDir,'DKE');
mkdir(dke_Path);

%% Read Files
fprintf('1: Reading Files\n');
%   Image
hdr = niftiinfo(dwi_Path);
dwi = niftiread(hdr);
dims = size(dwi);
fprintf('\tA:...loaded image\n');

%   BVEC and BVAL
bval = load(bval_Path);
fprintf('\tC:...loaded BVAL\n');
bvec = load(bvec_Path);
bval = round(bval, 1);
fprintf('\tD:...loaded BVEC\n');

%% Form Indexes
b0_idx = find(bval == 0);
b1_idx = find(bval == 1000);
b2_idx = find(bval == 2000);
fbi_idx = find(bval >= 2500);
rm_idx = cat(2, b0_idx, fbi_idx);

%% Handle Brain Mask and B0
if isfile(mask_Path)
    brainmask = logical(niftiread(mask_Path));
    fprintf('\tB:...loaded brainmask\n');
else
    brainmask = ones(dims(1:3));
end

if isfile(b0_Path)
    hdr_b0 = niftiinfo(b0_Path);
    b0 = niftiread(hdr_b0);
    fprintf('\tA:...loaded B0\n');
else
    b0 = nanmean(dwi(:,:,:,b0_idx),4);
    fprintf('\tA:...computed mean B0\n');
end

%% Remove B0s and FBI from all files
dwi(:,:,:,rm_idx) = [];
bval(rm_idx) = [];
bvec(:,rm_idx(2:end)) = [];

%% Concatenate with Designer B0
dwi = cat(4,b0,dwi);
b0_val = 0;
bval = cat(2,b0_val,bval);

%% Modify BVAL/BVEC
% Update path variables
dwi_Path = fullfile(dke_Path,'dke.nii');
bval_Path = fullfile(dke_Path,'dke.bval');
bvec_Path = fullfile(dke_Path,'dke.bvec');

fprintf('3: Writing Files\n');
for i = 1:size(dwi,4)
    dwi(:,:,:,i) = dwi(:,:,:,i) .* brainmask;
end
dwi(find(isnan(dwi))) = 0;      %remove nan
hdr.ImageSize = size(dwi);
fprintf('\tA:...writing image\n');
niftiwrite(dwi,dwi_Path,hdr);

fprintf('\tB:...writing BVAL\n');
dlmwrite(bval_Path,bval,',');
fprintf('\tC:...writing BVAL\n');
dlmwrite(bvec_Path,bvec,',');
b1_idx_new = find(bval == 1000); %indices have changed, update

[p,~,~] = fileparts(bvec_Path);
Gradient1 = bvec';
Gradient1 = Gradient1(b1_idx_new,:);
fprintf('\tD:...writing gradient\n');
save(fullfile(p,'gradient_dke.txt'),'Gradient1','-ASCII');

%% Create DKE Parameter File
fprintf('4: Creating DKE parameter files\n');
fid=fopen('dke_parameters.txt'); %Original file
fout=fullfile(dke_Path,'dke_parameters.txt');% new file

fidout=fopen(fout,'w');

while(~feof(fid))
    s=fgetl(fid);
    s=strrep(s,'dir-sub-changeme',dke_Path); %s=strrep(s,'A201', subject_list{i}) replace subject
    s=strrep(s,'ndir = ndir-changeme',sprintf('ndir = %d',length(b1_idx_new)));
    s=strrep(s,'bval = bval-changeme',sprintf('bval = [%s]', num2str(unique(bval))));
    s=strrep(s,'fn-gradients-changeme',fullfile(dke_Path,'gradient_dke.txt'));
    s=strrep(s,'fwhm_img = res-changeme',sprintf('fwhm_img = 0 * [%s]', num2str(hdr.PixelDimensions(1:3),3)));
    fprintf(fidout,'%s\n',s);
end
fclose(fid);
fclose(fidout);

%% Create FT Parameter File
fprintf('5: Creating FT parameter files\n');
fid=fopen('ft_parameters.txt'); %Original file
fout=fullfile(dke_Path,'ft_parameters.txt');% new file

fidout=fopen(fout,'w');

while(~feof(fid))
    s=fgetl(fid);
    s=strrep(s,'dir-sub-changeme',dke_Path); %s=strrep(s,'A201', subject_list{i}) replace subject
    s=strrep(s,'dir-trk-changeme',mask_Path); %s=strrep(s,'A201', subject_list{i}) replace subject
    s=strrep(s,'dir-seed-changeme',mask_Path); %s=strrep(s,'A201', subject_list{i}) replace subject
    fprintf(fidout,'%s\n',s);
end
fclose(fid);
fclose(fidout);
fprintf('.....Completed.....\n');
