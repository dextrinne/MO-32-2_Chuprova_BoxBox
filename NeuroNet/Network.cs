namespace MO_32_2_Chuprova_BoxBox.NeuroNet
{
    class Network
    {
        // Все слои сети
        private InputLayer input_Layer = null;
        private HiddenLayer hidden_Layer1;
        private HiddenLayer hidden_Layer2;
        private OutputLayer output_Layer;

        // Конфигурация сети
        private ActivationType hiddenActivation = ActivationType.Tanh;
        private double dropoutRate = 0.3; // 30% по умолчанию

        private double[] fact = new double[10]; // Массив фактического выхода сети
        private double[] e_error_avr; // Среднее значение энергии ошибки эпохи обучения
        private double[] train_accuracy; // Точность на обучающей выборке

        // Свойства
        public double[] Fact { get => fact; }
        public double[] E_error_avr { get => e_error_avr; set => e_error_avr = value; }
        public double[] Train_accuracy { get => train_accuracy; set => train_accuracy = value; }
        public ActivationType HiddenActivation { get => hiddenActivation; set => hiddenActivation = value; }
        public double DropoutRate { get => dropoutRate; set => dropoutRate = value; }

        // Конструктор с настройками
        public Network(ActivationType activationType = ActivationType.Tanh, double dropout = 0.3)
        {
            hiddenActivation = activationType;
            dropoutRate = dropout;

            // Инициализируем слои с выбранными параметрами
            hidden_Layer1 = new HiddenLayer(72, 15, NeuronType.Hidden, nameof(hidden_Layer1), hiddenActivation, dropoutRate);
            hidden_Layer2 = new HiddenLayer(32, 72, NeuronType.Hidden, nameof(hidden_Layer2), hiddenActivation, dropoutRate);
            output_Layer = new OutputLayer(10, 32, NeuronType.Output, nameof(output_Layer), ActivationType.Linear, 0.0);
        }

        // Прямой проход сети
        public void ForwardPass(Network net, double[] netInput, bool isTraining = false)
        {
            net.hidden_Layer1.SetData(netInput, isTraining);
            net.hidden_Layer1.Recognize(null, net.hidden_Layer2, isTraining);
            net.hidden_Layer2.Recognize(null, net.output_Layer, isTraining);
            net.output_Layer.Recognize(net, null, isTraining);
        }

        // Обучение нейросети
        public void Train(Network net)
        {
            net.input_Layer = new InputLayer(NetworkMode.Train);
            int epoches = 15;
            double tmpSumError;
            double[] errors;
            double[] temp_gsums1;
            double[] temp_gsums2;
            int epochCorrectPredictions = 0;
            int epochTotalPredictions = 0;

            e_error_avr = new double[epoches];
            train_accuracy = new double[epoches];

            for (int k = 0; k < epoches; k++)
            {
                e_error_avr[k] = 0;
                epochCorrectPredictions = 0;
                epochTotalPredictions = 0;

                //net.input_Layer.Shuffling_Array_Rows(net.input_Layer.Trainset);

                for (int i = 0; i < net.input_Layer.Trainset.GetLength(0); i++)
                {
                    double[] tmpTrain = new double[15];
                    for (int j = 0; j < tmpTrain.Length; j++)
                    {
                        tmpTrain[j] = net.input_Layer.Trainset[i, j + 1];
                    }

                    // Прямой проход с флагом обучения (для Dropout)
                    ForwardPass(net, tmpTrain, true);

                    // Вычисление точности
                    int predictedClass = GetPredictedClass(net.fact);
                    int actualClass = (int)net.input_Layer.Trainset[i, 0];
                    if (predictedClass == actualClass)
                    {
                        epochCorrectPredictions++;
                    }
                    epochTotalPredictions++;

                    // Вычисление ошибки
                    tmpSumError = 0;
                    errors = new double[net.fact.Length];
                    for (int x = 0; x < errors.Length; x++)
                    {
                        if (x == net.input_Layer.Trainset[i, 0])
                        {
                            errors[x] = 1.0 - net.fact[x];
                        }
                        else
                        {
                            errors[x] = -net.fact[x];
                        }
                        tmpSumError += errors[x] * errors[x] / 2;
                    }
                    e_error_avr[k] += tmpSumError / errors.Length;

                    // Обратный проход и коррекция весов
                    temp_gsums2 = net.output_Layer.BackwardPass(errors);
                    temp_gsums1 = net.hidden_Layer2.BackwardPass(temp_gsums2);
                    net.hidden_Layer1.BackwardPass(temp_gsums1);
                }

                e_error_avr[k] /= net.input_Layer.Trainset.GetLength(0);
                train_accuracy[k] = (double)epochCorrectPredictions / epochTotalPredictions * 100;
            }

            net.input_Layer = null;

            // Запись скорректированных весов
            net.hidden_Layer1.WeightsInitialize(MemoryMode.SET, nameof(hidden_Layer1) + "_memory.csv");
            net.hidden_Layer2.WeightsInitialize(MemoryMode.SET, nameof(hidden_Layer2) + "_memory.csv");
            net.output_Layer.WeightsInitialize(MemoryMode.SET, nameof(output_Layer) + "_memory.csv");
        }

        // Тестирование (без Dropout)
        public void Test(Network net)
        {
            net.input_Layer = new InputLayer(NetworkMode.Test);
            int epoches = 2;
            double tmpSumError;
            double[] errors;

            e_error_avr = new double[epoches];

            for (int k = 0; k < epoches; k++)
            {
                e_error_avr[k] = 0;
                net.input_Layer.Shuffling_Array_Rows(net.input_Layer.Testset);

                for (int i = 0; i < net.input_Layer.Testset.GetLength(0); i++)
                {
                    double[] tmpTrain = new double[15];
                    for (int j = 0; j < tmpTrain.Length; j++)
                    {
                        tmpTrain[j] = net.input_Layer.Testset[i, j + 1];
                    }

                    // Прямой проход без Dropout (isTraining = false)
                    ForwardPass(net, tmpTrain, false);

                    // Вычисление ошибки
                    tmpSumError = 0;
                    errors = new double[net.fact.Length];
                    for (int x = 0; x < errors.Length; x++)
                    {
                        if (x == net.input_Layer.Testset[i, 0])
                        {
                            errors[x] = 1.0 - net.fact[x];
                        }
                        else
                        {
                            errors[x] = -net.fact[x];
                        }
                        tmpSumError += errors[x] * errors[x] / 2;
                    }
                    e_error_avr[k] += tmpSumError / errors.Length;
                }

                e_error_avr[k] /= net.input_Layer.Testset.GetLength(0);
            }

            net.input_Layer = null;
        }

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
    }
}