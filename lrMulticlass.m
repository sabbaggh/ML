clc
clear all
close all
warning off all
%se leen los datos del dataset
datos = csvread('6 class csv.csv');
%se acomodan
datos = sortrows(datos,5);
%se leen los datos del archivo de pruebas
pruebass = csvread('pruebas.csv');
%Se meten a un vector
xpruebas = pruebass(:,:)'; 

numClases = 4;
vectoresSolucion = zeros(4,numClases-1);


for i =1:numClases-1
    datos = calcVector(datos,i);
    x=datos(:,1:4)';
    y=datos(:,5);
    indices = ~ismember(y, i);
    indices2 = ismember(y,i);
    y(indices) = -1;
    y(indices2) = 1;
    alfa = 0.000001;
    ep = 1;
    w0 = randn(4,1);
    vectoresSolucion(:,i) = calcGradDesc(alfa,w0,x,y,ep);
end




function nuevoVector = calcVector(datos,actual)
    indices = ~ismember(datos(:,5),actual-1);
    nuevoVector = datos(indices,:);
end

function wn = calcGradDesc(alfa,w0,x,y,ep)
    count = 0;
    while ep > 1e-6
        grad = calcGrad(x,y,w0);
        wn = w0 - alfa*grad;
        ep = sqrt((wn-w0)'*(wn-w0));
        w0 = wn;
        count = count+1;
    end
    fprintf('Iteraciones %d\n', count);
end


function gradiente = calcGrad(x,y,w)
    [m,n] = size(y);
    suma = 0;
    for i = 1:m
        arriba = -y(i)*x(:,i);
        abajo = 1 + exp(y(i)*w'*x(:,i));
        total = arriba/abajo;
        suma = suma + total;
    end
    gradiente =(1/m)*suma;
end

function precision = calcPrecision(xPrueba)
    [n, m] = size(xPrueba);
    %xPrueba = xPrueba';
    wT = wn';
    correcto = 0;
    for i =1:m
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
