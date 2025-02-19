clc
clear all
close all

% Based on M2 data
%% Read data file 

opts = detectImportOptions("Data_Up.csv");
opts = setvartype(opts,"double");

M2_table=readtable('Data_Up.csv',opts);
M2_array=table2array(M2_table);
M2_wavelength=1e7*(1./M2_array(1,3:end));

% Get class and biomarker data 
M2_class=(M2_array(2:end,1));
M2_bmd=M2_array(2:end,2);
M2_array_data=M2_array(2:end,:);

% clean up the missing data
cleaned_data=M2_array_data(~isnan(M2_class),:);
row_nan=any(isnan(cleaned_data),2);
cleaned_data=cleaned_data(~row_nan,:);
M2_data=cleaned_data(:,3:end);

% clean up biomarker data
M2_class_cleaned=M2_class(~isnan(M2_class));
M2_class_cleaned=M2_class_cleaned(~row_nan,:)+1;
M2_bmd_cleaned=M2_bmd(~isnan(M2_bmd));
M2_bmd_cleaned=M2_bmd_cleaned(~row_nan,:);

%% select data range

range={200:1000;1000:2000;1500:2000}; % range selection for wavelength
id=1; % set id for which range should be selected

%% Exploratory data analysis
% Normalize and PCA
M2_data_norm=zscore(M2_data(:,range{id}));
[coeff,score,latent,~,explained_var]=pca(M2_data_norm);

%Plot PCA
figure;
gscatter(score(:,1),score(:,2),M2_class_cleaned, 'rgb', 'o', 8)

figure
plot(M2_wavelength(range{id}),coeff(:,1))
hold on
plot(M2_wavelength(range{id}),coeff(:,2))

% Determine Explained variance and important components 
explained_variance=cumsum(explained_var);
num_components=find(explained_variance <99.5);


%% Regression

PCA_data=score(:,1:2); 
% Xregx=[M2_data_norm(1:end,1339+2) M2_data_norm(1:end,1298+2) M2_data_norm(1:end,1963+2)  M2_data_norm(1:end,288+2) M2_data_norm(1:end,50+2) M2_data_norm(1:end,60+2)];

% cross validation 
for i=1:100
cv=cvpartition(size(M2_data_norm,1),'Holdout',.8);

train_data=PCA_data(training(cv),:);
test_data=PCA_data(test(cv),:);

train_target=M2_bmd_cleaned(training(cv),:);
test_target=M2_bmd_cleaned(test(cv),:);
% 
% Xreg=Xregx(training(cv),:);
% Xreg_t=Xregx(test(cv),:);
% model_x=fitlm(Xreg,train_target); % Linear Model


model=fitlm(train_data,train_target);


y_target=predict(model,test_data);

% Calculate RMSE
error(i)=mean(abs(y_target-test_target));

end


Mean_abs_error=mean(abs(error));
disp(Mean_abs_error)