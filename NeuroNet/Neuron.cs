using static System.Math;

namespace MO_32_2_Chuprova_BoxBox.NeuroNet
{
    // Функция активации: Гиперболический тангенс, 15-72-32-10

    class Neuron
    {
        // Поля
        private NeuronType type; // Тип нейрона
        private double[] weights; // Его веса
        private double[] inputs; // Его входы
        private double output; // Его выход
        private double derivative; // Производная

        // Свойства
        public double[] Weights { get => weights; set => weights = value; }
        public double[] Inputs { get => inputs; set => inputs = value; }
        public double Output { get => output; }
        public double Derivative { get => derivative; }

        // Конструктор
        public Neuron(double[] memoryWeights, NeuronType typeNeuron)
        {
            type = typeNeuron;
            weights = memoryWeights;
        }

        // Метод активации нейрона (нелинейные преобразования входного сигнала)
        public void Activator(double[] i)
        {
            inputs = i; // Передача вектора входного сигнала в массив входных данных

            double sum = weights[0]; // Аффиное преобразование через смещение

            for (int j = 0; j < inputs.Length; j++) // Цикл вычисления индуцирования
            {
                sum += inputs[j] * weights[j + 1]; // Линейные преобразования входных данных
            }

            switch (type)
            {
                case NeuronType.Hidden:
                    output = Tangens(sum);
                    derivative = Tangens_Derivativator(sum);
                    break;

                case NeuronType.Output: // Функция soft-max
                    // Выходные нейроны используют exp для Softmax (нормализация в слое)
                    // Производная для Softmax вычисляется на уровне слоя
                    output = sum;
                    derivative = 1.0; // Заглушка, реальная производная вычисляется в OutputLayer
                    break;
            }
        }

        // Функция активации нейрона (гиперболический тангенс)
        private double Tangens(double sum)
        {
            double outp = (Exp(sum) - Exp(-sum)) / (Exp(sum) + Exp(-sum));
            return outp;
        }

        // Производная гиперболического тангенса
        private double Tangens_Derivativator(double sum)
        {
            double th = Tangens(sum);
            return 1 - th * th; // Более стабильная формула
        }
    }
}
