function ExampleRBM
    clear, clc;
    format longg;
    
    % Инициализируем количество эпох, learning rate и количество шагов
    % алгоритма CD-k
    epoch_count = 1;
    disp("Количество эпох: " + epoch_count);
    
    learning_rate = 0.5;
    disp("Learning rate: " + learning_rate);
    
    CD_k = 2;
    disp("Количество шагов алгоритма CD-k: " + CD_k);
    
    % Обучающая выборка
    S = [0 1 0; 0 0 1; 1 1 0];
    disp("Обучающая выборка:");
    disp(S);
    
    % Инициализируем начальные значения матрицы весов и векторов сдвигов
    disp("Инициализация начальных значений матрицы весов и векторов сдвигов");
    disp("Вектор сдвигов 'a':");
    a = [0; 0; 0];
    disp(a);
    
    disp("Вектор сдвигов 'b':");
    b = [0; 0];
    disp(b);
    
    disp("Матрица весов 'W':");
    W = [1 -1; -2 2; 1 -1];
    disp(W);
    
    Nv = size(W, 1);
    Nh = size(W, 2);
    disp("Количество нейронов видимого слоя: " + Nv);
    disp("Количество нейронов скрытого слоя: " + Nh);
    disp("(В данной программе они определяются размером матрицы 'W')" + newline);
    
    % Количество векторов в обучающей выборке
    v_count = size(S, 2);
    disp("Количество векторов в обучающей выборке: " + v_count + newline);
    
    % Цикл по эпохам
    for epoch = 1:epoch_count
        disp(epoch + " эпоха");
        % Цикл по векторам обучающей выборки
        for v_index = 1:v_count
            % Инициализируем текущий вектор из обучающей выборки
            disp("Номер вектора из обучающей выборки: " + v_index);
            
            v0 = S(:, v_index);
            disp("Вектор v0:");
            disp(v0);
            
            v = v0;
            % Цикл по шагам алгоритма CD-k
            for k = 1:CD_k
                disp("Текущий шаг CD-k: " + k + newline);
                
                % Вычисляем вероятность p(h = 1|v)
                p_h_v = Sigmoid(b + W' * v);
                disp("Вероятность p(h" + (k - 1) + " = 1|v" + (k - 1) + "):");
                disp(p_h_v');
                
                % Cэмплируем вероятность (h ~ p(h = 1|v))
                h = Sampling(p_h_v);
                disp("Вектор h" + (k - 1) + " после сэмплирования:");
                disp(h);
                
                % Вычисляем вероятность p(v = 1|h)
                p_v_h = Sigmoid(a + W * h);
                disp("Вероятность p(v" + k + " = 1|h" + (k - 1) + "):");
                disp(p_v_h');
                
                % Cэмплируем вероятность (h ~ p(v = 1|h))
                v = Sampling(p_v_h);
                disp("Вектор v" + k + " после сэмплирования:");
                disp(v);
            end
            disp(newline + "Теперь обновим значения матрицы весов и векторов сдвигов" + newline);
            % Вычисляем вероятности p(h = 1|v0) и p(h = 1|vk)
            
            p_h_v0 = Sigmoid(b + W' * v0);
            disp("Вероятность p(h = 1|v0):");
            disp(p_h_v0');
            
            p_h_vk = Sigmoid(b + W' * v);
            disp("Вероятность p(h = 1|vk):");
            disp(p_h_vk');
            
            % Двойной цикл для обновления значений в матрице весов 'W' и
            % векторе сдвигов 'a'
            for i = 1:Nv
                % Пересчитываем значения вектора сдвига 'a'
                a(i) = a(i) + learning_rate * (v0(i) - v(i));
                for j = 1:Nh
                    % Пересчитываем значения матрицы весов 'W'
                    W(i, j) = W(i, j) + learning_rate * (p_h_v0(j) * v0(i) - p_h_vk(j) * v(i));
                end
            end
            for j = 1:Nh
                % Пересчитываем значения вектора сдвига 'b'
                b(j) = b(j) + learning_rate * (p_h_v0(j) - p_h_vk(j));
            end
            
            disp("Новый вектор сдвигов 'a':");
            disp(a);

            disp("Новый вектор сдвигов 'b':");
            disp(b);

            disp("Новая матрица весов 'W':");
            disp(W);
        end
    end
    disp("Обучение RBM завершено" + newline);
    
    % Тестовая выборка
    Test = [1 0 1 0; 0 1 1 0; 1 0 0 1];
    disp("Тестовая выборка:");
    disp(Test);
    
    % Количество векторов в тестовой выборке
    v_count = size(Test, 2);
    disp("Количество векторов в тестовой выборке: " + v_count + newline);
    
    % Массив ошибок: 1 - вектора различны, 0 - в противном случае
    Errors = zeros(1, v_count)';
    
    % Массив предсказанных векторов
    Predict = zeros(size(Test, 1), size(Test, 2));
    
    % Цикл по векторам тестовой выборки
    for v_index = 1:v_count
        v0 = Test(:, v_index);
        v = v0;
        % Цикл по шагам алгоритма CD-k
        for k = 1:CD_k
            p_h_v = Sigmoid(b + W' * v);
            h = Sampling(p_h_v);
            p_v_h = Sigmoid(a + W * h);
            v = Sampling(p_v_h);
        end
        Predict(:,v_index) = v;
        Errors(v_index) = CheckVectors(v0, v);
    end
    disp("Исходные вектора (по столбцам):");
    disp(Test);
    disp("Предсказанные вектора (по столбцам):");
    disp(Predict);
    disp("Посмотрим на ошибки для векторов тестовой выборки (0 - ошибки нет, 1 - ошибка есть):");
    for v_index = 1:v_count
        disp("Ошибка для " + v_index + " вектора тестовой выборки: " + Errors(v_index));
    end
end

% Вычисляет сигмоидную функцию от каждой компоненты вектора X
function Ans = Sigmoid(X)
    Ans = zeros(1, length(X));
    for i = 1:length(X)
        Ans(i) = 1/(1 + exp(-X(i)));      
    end
end

% Сэмплирует вероятности по следующему принципу: 
% если вероятность больше 0.5, то компонента принимает значение 1,
% и 0 в противном случае
function Ans = Sampling(X)
    Ans = zeros(1, length(X));
    for i = 1:length(X)
        if (X(i) > 0.5)
            Ans(i) = 1;
        else
            Ans(i) = 0;
        end
    end
    Ans = Ans';
end

% Проверяет 2 вектора на отличия: 1 - вектора  различны, 
% 0 - вектора совпали
function Ans = CheckVectors(X1, X2)
    for i = 1:length(X1)
        if (X1(i) ~= X2(i))
            Ans = 1;
            return
        end     
    end
    Ans = 0;
end
