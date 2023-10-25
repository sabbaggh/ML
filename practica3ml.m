clc
clear all
close all
warning off all
%se leen los datos del dataset
datos = csvread('6 class csv2.csv');
%se acomodan
%datos = sortrows(datos,5);
[m,n]=size(datos);
pruebass = csvread('pruebas2.csv');
xpruebas = pruebass(:,:)';
numClases = 4;

lambda = 55;
alfa = 0.00000001;
count = 0;

for i = 1:numClases
    for j = i+1:numClases
        ep = 1;
        w0 = randn(4,1);
        nuevosDatos = crearNuevoVector(i,j,datos);
        x = nuevosDatos(:,1:4)';
        y = nuevosDatos(:,5);
        indices = ~ismember(y, i);
        indices2 = ismember(y,i);
        y(indices) = -1;
        y(indices2) = 1;
        while ep > 1e-6
            gradiente = calcGrad(x,y,w0,lambda);
            wn = w0 - alfa*gradiente;
            ep = sqrt((wn-w0)'*(wn-w0));
            w0 = wn;
            count = count+1;
        end
        fprintf('Iteraciones %d\n', count)
        
    end
end
test = prueba(xpruebas,wn);

%while ep > 1e-6
    %gradiente = calcGrad(x,y,w0,lambda);
   % wn = w0 - alfa*gradiente;
   % ep = sqrt((wn-w0)'*(wn-w0))
  %  w0 = wn;
 %   count = count+1;
%end
%fprintf('Iteraciones %d\n', count)
%test = prueba(xpruebas,wn);




%FUNCIONES A USAR
function grad = calcGrad(x, y, w0,lambda)
    [m,~] = size(y);
    grad = zeros(4,1);
    for i = 1:m
        decision = x(:, i) * w0';
        decision = y(i)*decision;
        if decision<1
            grad = grad - y(i)*x(:,i) + 2*lambda*w0;
        else
            grad = grad + 2*lambda*w0;
        end
    end
end

function nuevosDatos = crearNuevoVector(clase,actual,datos)
    clase1 = find(datos(:,5)==clase);
    clase2 = find(datos(:,5) == actual);
    clase11 = datos(clase1,:);
    clase22 = datos(clase2,:);
    nuevosDatos = [clase11;clase22];
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
        if h > 0 && (xPrueba(5,i) ==3)
            correcto = correcto+1;
            disp('Correcto')
        elseif h < 0 && (xPrueba(5,i) ==4)
            correcto = correcto+1;
            disp('Correcto')
        end
    end
    precision = 100*(correcto/m)
    fprintf('El modelo obtuvo una precision del %d%% en los datos de prueba\n',precision)
end
