using System;
using static System.Math;

namespace MO_32_2_Chuprova_BoxBox.NeuroNet
{
    class Neuron
    {
        // Поля
        private NeuronType type; // Тип нейрона
        private ActivationType activationType; // Тип функции активации
        private double[] weights; // Его веса
        private double[] inputs; // Его входы
        private double output; // Его выход
        private double derivative; // Производная

        private double dropoutProbability; // Вероятность отключения (для Dropout)
        private bool isDropped; // Флаг отключения нейрона

        // Свойства
        public double[] Weights { get => weights; set => weights = value; }
        public double[] Inputs { get => inputs; set => inputs = value; }
        public double Output { get => output; }
        public double Derivative { get => derivative; }

        public ActivationType ActivationType { get => activationType; set => activationType = value; }
        public double DropoutProbability { get => dropoutProbability; set => dropoutProbability = value; }
        public bool IsDropped { get => isDropped; set => isDropped = value; }

        // Конструктор
        public Neuron(double[] memoryWeights, NeuronType typeNeuron, ActivationType actType = ActivationType.Tanh, double dropoutProb = 0.0)
        {
            type = typeNeuron;
            weights = memoryWeights;
            activationType = actType;
            dropoutProbability = dropoutProb;
            isDropped = false;
        }

        private static Random randd = new Random();

        // Метод активации нейрона (нелинейные преобразования входного сигнала)
        public void Activator(double[] i, bool isTraining = false)
        {
            inputs = i; // Передача вектора входного сигнала в массив входных данных

            // Применение Dropout во время обучения
            if (isTraining && type == NeuronType.Hidden && dropoutProbability > 0)
            {
                isDropped = randd.NextDouble() < dropoutProbability;
                if (isDropped)
                {
                    output = 0;
                    derivative = 0;
                    return;
                }
            }
            else
            {
                isDropped = false;
            }

            double sum = weights[0]; // Аффинное преобразование через смещение

            for (int j = 0; j < inputs.Length; j++) // Цикл вычисления индуцирования
            {
                sum += inputs[j] * weights[j + 1]; // Линейные преобразования входных данных
            }

            switch (activationType)
            {
                case ActivationType.Sigmoid:
                    output = Sigmoid(sum);
                    derivative = SigmoidDerivative(sum);
                    break;
                case ActivationType.Tanh:
                    output = Tanh(sum);
                    derivative = TanhDerivative(sum);
                    break;
                case ActivationType.ReLU:
                    output = ReLU(sum);
                    derivative = ReLUDerivative(sum);
                    break;
                case ActivationType.LeakyReLU:
                    output = LeakyReLU(sum);
                    derivative = LeakyReLUDerivative(sum);
                    break;
                case ActivationType.UnitStep:
                    output = UnitStep(sum);
                    derivative = UnitStepDerivative(sum);
                    break;
                case ActivationType.Linear:
                    output = Linear(sum);
                    derivative = LinearDerivative();
                    break;
                default:
                    output = Tanh(sum);
                    derivative = TanhDerivative(sum);
                    break;
            }
        }

        // Функции активации
        // Сигмоидальная (логистическая) функция
        private double Sigmoid(double sum)
        {
            return 1.0 / (1.0 + Exp(-sum));
        }
        private double SigmoidDerivative(double sum)
        {
            //return y * (1 - y);
            return Exp(-sum) / Pow((Exp(-sum) + 1.0), 2);
        }

        // Гиперболический тангенс
        private double Tanh(double x)
        {
            return (Exp(x) - Exp(-x)) / (Exp(x) + Exp(-x));
        }
        private double TanhDerivative(double sum)
        {
            double th = Tanh(sum);
            return 1 - th * th; // Более стабильная формула
        }

        // ReLU
        private double ReLU(double x)
        {
            return x > 0 ? x : 0;
        }
        private double ReLUDerivative(double x)
        {
            return x > 0 ? 1 : 0;
        }

        // Leaky ReLU
        private double LeakyReLU(double sum)
        {
            //return x > 0 ? x : 0.01 * x;
            return sum >= 0 ? sum : 0.01 * sum;
        }
        private double LeakyReLUDerivative(double sum)
        {
            //return x > 0 ? 1 : 0.01;
            return sum < 0 ? 0.01 : 1.0;
        }

        // Единичный скачок
        private double UnitStep(double x, double alpha = 100.0)
        {
            // Чем больше alpha, тем круче переход
            // Практически это очень крутая сигмоида
            return 1.0 / (1.0 + Math.Exp(-alpha * x));
        }
        private double UnitStepDerivative(double x, double alpha = 100.0)
        {
            double y = UnitStep(x, alpha);
            return alpha * y * (1 - y);
        }

        // Линейная функция
        private double Linear(double x)
        {
            return x;
        }
        private double LinearDerivative()
        {
            return 1;
        }
    }
}