clc
clear all
close all
warning off all

%se leen los datos del dataset
datos = csvread('6 class csv.csv');
%se acomodan
datos = sortrows(datos,5);
[m,n]=size(datos);
pruebass = csvread('pruebas.csv');
xpruebas = pruebass(:,:)'; 
y = zeros(m,1);
%Creamos el vector de etiquetas y
y = datos(:,5);
%Se crea la matriz de los vectores de caracteristicas x
x = datos(:,1:4)';
xT = x';
grid on
hold on
%view(3)
scatter(x(4,:),y(:,1))
xlabel('Temperatura')
ylabel('Etiquetas')
indices = ~ismember(y, [1, 2]);
%se cambian todos los indices de y que no sean 1 para tener solamente dos
%clases
y(indices) = -1;

matrizEig = xT*x;
eigenvalores = eig(matrizEig);
maxEig = max(eigenvalores);
%Creacion del vector inicial w con valores random
w0 = randn(4,1);
ep = 1;
alfa = 0.0000001;
count = 0;

while ep > 1e-6
    grad = calcGrad(x,y,w0);
    wn = w0 - alfa*grad
    ep = sqrt((wn-w0)'*(wn-w0));
    w0 = wn;
    count = count+1
end
test = prueba(datos,wn);


%FUNCIONES A USAR
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



function diagonal = calcdiag(x,w,xT)
    [m,n] = size(xT);
    wT = w'
    daigonales = zeros(1,m);
    for i = 1:m
        xdd = exp(-wT*x(:,i))%*(1-(1/(1+exp(-w'*x(:,i)))))
    end
    %diagonal = diag(daigonales)

end

function precision = prueba(xPrueba,wn)
    [n, m] = size(xPrueba);
    xPrueba = xPrueba';
    wT = wn';
    correcto = 0;
    for i =1:m
        h = wT*xPrueba(1:4,i);
        if h > 0 && (xPrueba(5,i) ==1 || xPrueba(5,i) ==2)
            correcto = correcto+1;
        elseif h < 0 && (xPrueba(5,i) ==4 || xPrueba(5,i) ==5)
            correcto = correcto+1;
        end
    end
    precision = 100*(correcto/m)
    fprintf('El modelo obtuvo una precision del %d%% en los datos de prueba\n',precision)
end
    
