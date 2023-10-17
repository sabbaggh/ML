clc
clear all
close all
warning off all
%se leen los datos del dataset
datos = csvread('6 class csv.csv');
%se acomodan
datos = sortrows(datos,5)

numClases = 4;


for i =1:numClases-1
    datos = calcVector(datos,i)
end




function nuevoVector = calcVector(datos,actual)
    indices = ~ismember(datos(:,5),actual-1);
    nuevoVector = datos(indices,:);
end
