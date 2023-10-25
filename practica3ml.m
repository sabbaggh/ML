clc
clear all
close all
warning off all
%se leen los datos del dataset
datos = csvread('6 class csv2.csv');
%se acomodan
%datos = sortrows(datos,5);
[m,n]=size(datos);
pruebass = csvread('6 class csv2.csv');
xpruebas = pruebass(:,:)'; 
y = zeros(m,1);
%Creamos el vector de etiquetas y
y = datos(:,5);
%Se crea la matriz de los vectores de caracteristicas x
x = datos(:,1:4)';
%Se buscan los indices que no tengan etiqueta 1 o 2 y los que tengan
%etiqueta 1 o 2, esto para juntar dos clases, la 1 con 2 y la 3 con 4
indices = ~ismember(y, [1, 2]);
indices2 = ismember(y,[1, 2]);
%se cambian las etiquetas de acuerdo a los indices que obtuvimos
y(indices) = -1;
y(indices2) = 1;

lambda = 50;
w0 = zeros(4,1);
ep = 1;
alfa = 0.000000001;
count = 0;
b=0;

while ep > 1e-6
%for i = 1:10000
    %grad = calcGrad(x,y,w0,lambda);
    [m,n] = size(y);
    grad = zeros(4,1);
    gradb=0;
    for i = 1:m
        decision = x(:, i) * w0' + b;
        if decision<1
            grad = grad - y(i)*x(:,i) + 2*lambda*w0;
            gradb = gradb - y(i);
        else
            grad = grad + 2*lambda*w0;
        end
    end
    %nomrCuad = norm(w)^2;
    gradiente = grad; %+ lambda*nomrCuad;
    wn = w0 - alfa*grad;
    b=b-alfa*gradb;
    ep = sqrt((wn-w0)'*(wn-w0))
    w0 = wn;
    count = count+1;
end
fprintf('Iteraciones %d\n', count)
test = prueba(xpruebas,wn);




%FUNCIONES A USAR
function gradiente = calcGrad(x, y, w,lambda)
    [m,n] = size(y);
    grad = zeros(4,1);
    gradb=0;
    for i = 1:m
        decision = x(i, :) * w' + b;
        if decision<1
            grad = grad - y(i)*x(:,i) + 2*lambda*w;
            gradb = gradb - y(i);
        else
            grad = grad + 2*lambda*w;
        end
    end
    nomrCuad = norm(w)^2;
    gradiente = grad; %+ lambda*nomrCuad;
end


%Funcion para calcular la precision del modelo
function precision = prueba(xPrueba,wn)
    [~, m] = size(xPrueba);
    %Se hace la transpuesta del wn obtenido
    wT = wn';
    %Contador para llevar la cuenta de predicciones correctas
    correcto = 0;
    %Se ira recorriendo los vectores de prueba 
    for i =1:m
        %Se calcula la hipotesis
        h = wT*xPrueba(1:4,i)
        if h > 0 && (xPrueba(5,i) ==1 || xPrueba(5,i) ==2)
            correcto = correcto+1;
        elseif h < 0 && (xPrueba(5,i) ==3 || xPrueba(5,i) ==4)
            correcto = correcto+1;
        end
    end
    precision = 100*(correcto/m)
    fprintf('El modelo obtuvo una precision del %d%% en los datos de prueba\n',precision)
end
