namespace MO_32_2_Chuprova_BoxBox.NeuroNet
{
    // Функция активации: Гиперболический тангенс, 15-72-32-10
    class Network
    {
        // Все слои сети
        private InputLayer input_Layer = null;
        private HiddenLayer hidden_Layer1 = new HiddenLayer(72, 15, NeuronType.Hidden, nameof(hidden_Layer1));
        private HiddenLayer hidden_Layer2 = new HiddenLayer(32, 72, NeuronType.Hidden, nameof(hidden_Layer2));
        private OutputLayer output_Layer = new OutputLayer(10, 32, NeuronType.Output, nameof(output_Layer));

        private double[] fact = new double[10]; // Массив фактического выхода сети
        private double[] e_error_avr; // Среднее значение энергии ошибки эпохи обучения

        // Новое поле для хранения точности
        private double[] train_accuracy; // Точность на обучающей выборке

        // Свойства
        public double[] Fact { get => fact; } // Массив фактического выхода сети

        // Среднее значение энергии ошибки эпохи обучения
        public double[] E_error_avr { get => e_error_avr; set => e_error_avr = value; }

        // Свойство для точности
        public double[] Train_accuracy { get => train_accuracy; set => train_accuracy = value; }

        // Конструктор
        public Network() { }

        // Прямой проход сети
        public void ForwardPass(Network net, double[] netInput)
        {
            net.hidden_Layer1.Data = netInput;
            net.hidden_Layer1.Recognize(null, net.hidden_Layer2);
            net.hidden_Layer2.Recognize(null, net.output_Layer);
            net.output_Layer.Recognize(net, null);
        }

        // Метод для определения предсказанного класса (класс с максимальной вероятностью)
        private int GetPredictedClass(double[] outputs)
        {
            int predictedClass = 0;
            double maxOutput = outputs[0];

            for (int i = 1; i < outputs.Length; i++)
            {
                if (outputs[i] > maxOutput)
                {
                    maxOutput = outputs[i];
                    predictedClass = i;
                }
            }

            return predictedClass;
        }

        // Обучение нейросети
        public void Train(Network net)
        {
            net.input_Layer = new InputLayer(NetworkMode.Train); // Инициализация входного слоя
            int epoches = 15; // Количество эпох обучения (сами настраиваем (до 20))
            double tmpSumError; // Временная переменная суммы ошибок (столько же сколько и нейронов)
            double[] errors; //  Вектор (массив) сигнала ошибки выходного слоя
            double[] temp_gsums1; // Вектор градиента 1-ого скрытого слоя
            double[] temp_gsums2; // Вектор градиента 2-ого скрытого слоя
            int epochCorrectPredictions = 0;
            int epochTotalPredictions = 0;

            e_error_avr = new double[epoches];
            train_accuracy = new double[epoches]; // Инициализация массива точности

            for (int k = 0; k < epoches; k++) // Перебор эпох обучения
            {
                e_error_avr[k] = 0;

                // Для расчета точности
                epochCorrectPredictions = 0;
                epochTotalPredictions = 0;

                net.input_Layer.Shuffling_Array_Rows(net.input_Layer.Trainset); // Перетасовка обучающей выборки

                for (int i = 0; i < net.input_Layer.Trainset.GetLength(0); i++)
                {
                    double[] tmpTrain = new double[15]; // Обучающий образ
                    for (int j = 0; j < tmpTrain.Length; j++)
                    {
                        tmpTrain[j] = net.input_Layer.Trainset[i, j + 1]; 
                    }
                    // Прямой проход обучающего образа
                    ForwardPass(net, tmpTrain);

                    // ВЫЧИСЛЕНИЕ ТОЧНОСТИ ДЛЯ ОБУЧЕНИЯ
                    int predictedClass = GetPredictedClass(net.fact);
                    int actualClass = (int)net.input_Layer.Trainset[i, 0];

                    if (predictedClass == actualClass)
                    {
                        epochCorrectPredictions++;
                    }
                    epochTotalPredictions++;

                    // Вычисление ошибки по итерации
                    tmpSumError = 0; // Для каждого обучающего образа среднее значение ошибки этого образа равна 0 
                    errors = new double[net.fact.Length]; // Переопределение массива сигнала ошибки выходного слоя
                    for (int x = 0; x < errors.Length; x++)
                    {
                        if (x == net.input_Layer.Trainset[i, 0])
                        {
                            errors[x] = 1.0 - net.fact[x];
                        }
                        else
                        {
                            errors[x] = - net.fact[x]; // errors[x] = 0 - net.fact[x];
                        }
                        tmpSumError += errors[x] * errors[x] / 2;
                    }
                    e_error_avr[k] += tmpSumError / errors.Length; // Суммарное значение энергии ошибки k-ой ошибки

                    // Обратный проход и коррекция весов
                    temp_gsums2 = net.output_Layer.BackwardPass(errors);
                    temp_gsums1 = net.hidden_Layer2.BackwardPass(temp_gsums2);
                    net.hidden_Layer1.BackwardPass(temp_gsums1);
                }
                e_error_avr[k] /= net.input_Layer.Trainset.GetLength(0); // Среднее значение энергии ошибки одной эпохи
                
                // РАСЧЕТ ТОЧНОСТИ ДЛЯ ЭПОХИ
                train_accuracy[k] = (double)epochCorrectPredictions / epochTotalPredictions * 100;
            }

            net.input_Layer = null; // Обнуление слоя

            // Запись скорректированных весов
            net.hidden_Layer1.WeightsInitialize(MemoryMode.SET, nameof(hidden_Layer1) + "_memory.csv");
            net.hidden_Layer2.WeightsInitialize(MemoryMode.SET, nameof(hidden_Layer2) + "_memory.csv");
            net.output_Layer.WeightsInitialize(MemoryMode.SET, nameof(output_Layer) + "_memory.csv");
        }

        public void Test(Network net)
        {
            net.input_Layer = new InputLayer(NetworkMode.Test); // Инициализация входного слоя
            int epoches = 2; // Количество эпох обучения (сами настраиваем (от 2 до 5))
            double tmpSumError; // Временная переменная суммы ошибок (столько же сколько и нейронов)
            double[] errors; //  Вектор (массив) сигнала ошибки выходного слоя
            double[] temp_gsums1; // Вектор градиента 1-ого скрытого слоя
            double[] temp_gsums2; // Вектор градиента 2-ого скрытого слоя

            e_error_avr = new double[epoches];
            for (int k = 0; k < epoches; k++) // Перебор эпох обучения
            {
                e_error_avr[k] = 0;
                net.input_Layer.Shuffling_Array_Rows(net.input_Layer.Testset); // Перетасовка обучающей выборки

                for (int i = 0; i < net.input_Layer.Testset.GetLength(0); i++)
                {
                    double[] tmpTrain = new double[15]; // Обучающий образ
                    for (int j = 0; j < tmpTrain.Length; j++)
                    {
                        tmpTrain[j] = net.input_Layer.Testset[i, j + 1];
                    }
                    // Прямой проход обучающего образа
                    ForwardPass(net, tmpTrain);

                    // Вычисление ошибки по итерации
                    tmpSumError = 0; // Для каждого обучающего образа среднее значение ошибки этого образа равна 0 
                    errors = new double[net.fact.Length]; // Переопределение массива сигнала ошибки выходного слоя
                    for (int x = 0; x < errors.Length; x++)
                    {
                        if (x == net.input_Layer.Testset[i, 0])
                        {
                            errors[x] = 1.0 - net.fact[x];
                        }
                        else
                        {
                            errors[x] = -net.fact[x]; // errors[x] = 0 - net.fact[x];
                        }
                        tmpSumError += errors[x] * errors[x] / 2;
                    }
                    e_error_avr[k] += tmpSumError / errors.Length; // Суммарное значение энергии ошибки k-ой ошибки
                }
                e_error_avr[k] /= net.input_Layer.Testset.GetLength(0); // Среднее значение энергии ошибки одной эпохи
            }

            net.input_Layer = null; // Обнуление слоя

        }
    }
}
