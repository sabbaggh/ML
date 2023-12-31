clc
clear all
close all
warning off all
%se leen los datos del dataset
datos = csvread('6 class csv2.csv');
%se acomodan
datos = sortrows(datos,5);
[m,n]=size(datos);
pruebass = csvread('pruebas2.csv');
xpruebas = pruebass(:,:)'; 
y = zeros(m,1);
%Creamos el vector de etiquetas y
y = datos(:,5);
%Se crea la matriz de los vectores de caracteristicas x
x = datos(:,1:4)';
xT = x';
hold on
grid on
scatter(x(4,:),y(:,1),'go',MarkerFaceColor='g')
xlabel('Magnitud')
ylabel('Etiquetas')
%Se buscan los indices que no tengan etiqueta 1 o 2 y los que tengan
%etiqueta 1 o 2, esto para juntar dos clases, la 1 con 2 y la 3 con 4
indices = ~ismember(y, [1, 2]);
indices2 = ismember(y,[1, 2]);
%se cambian las etiquetas de acuerdo a los indices que obtuvimos
y(indices) = -1;
y(indices2) = 1;

%Creacion del vector inicial w con valores random
w0 = randn(4,1);
%Epsilon inicial
ep = 1;
%Se usa el mismo alfa de la practica anterior
alfa = 0.000001;
%Contador para ver el numero de iteraciones
count = 0;

%Se realiza el proceso del gradiente descendiente
while ep > 1e-6
    grad = calcGrad(x,y,w0);
    wn = w0 - alfa*grad;
    ep = sqrt((wn-w0)'*(wn-w0));
    w0 = wn;
    count = count+1;    
end
fprintf('Iteraciones %d\n', count)
test = prueba(xpruebas,wn);

%FUNCIONES A USAR
%Se hace una funcion para calcular el gradiente, esta se manda a llamar en
%cada iteracion del proceso de gradiente descendiente
function gradiente = calcGrad(x, y, w)
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

%Funcion para calcular la precision del modelo
function precision = prueba(xPrueba,wn)
    [n, m] = size(xPrueba);
    %Se hace la transpuesta del wn obtenido
    wT = wn';
    %Contador para llevar la cuenta de predicciones correctas
    correcto = 0;
    %Se ira recorriendo los vectores de prueba 
    for i =1:m
        %Se calcula la hipotesis y si es mayor a cero y ademas su etiqueta
        %es 1 o 2 se considera como correcta, si es menor a 0 la hipotesis
        %y ademas su etiqueta es 3 o 4 entonces se considera como correcta
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
    
