using System;
using System.IO;
using System.Windows.Forms;
using static System.Math;

namespace MO_32_2_Chuprova_BoxBox.NeuroNet
{
    abstract class Layer
    {
        // Модификатор protected работает аналогично private, но необходим для внутрииерархического использования членов
        // По умолчанию private

        // Поля
        protected string name_Layer;
        string pathDirWeights; // Путь к каталогу, где находится файл синаптических весов для нейросети
        string pathFileWeights; // Путь к файлу синаптических весов для нейросети
        protected int numofneurons; // Число нейронов текущего слоя
        protected int numofprevneurons; // Число нейронов предыдущего слоя
        protected const double learningrate = 0.031d; // Скорость обучения (подбираем сами: 041155 0.029383 0.031) - коэффициент обновления весов
        protected const double momentum = 0.2d; // Момент инерции (подбираем сами: 0.00465 0.2) - помогает избежать локальных минимумов
        protected double[,] lastdeltaweights; // Веса предыдущей итерации обучения
        protected Neuron[] neurons; // Массив нейронов текущего слоя


        // Свойства
        public Neuron[] Neurons { get => neurons; set => neurons = value; } // Массив нейронного слоя
        public double[] Data // Передача входных данных на нейроны слоя и активация нейрона
        {
            set
            {
                // Передаем входные данные каждому нейрону и активируем его
                for (int i = 0; i < numofneurons; i++)
                {
                    Neurons[i].Activator(value);
                }
            }
        }

        // Конструктор
        protected Layer(int non, int nopn, NeuronType nt, string nm_layer)
        {
            numofneurons = non; // Количество нейронов текущего слоя
            numofprevneurons = nopn; // Количество нейронов предыдущего слоя
            Neurons = new Neuron[non]; // Определение массива нейронов
            name_Layer = nm_layer;

            // Формирование путей для хранения весов
            pathDirWeights = AppDomain.CurrentDomain.BaseDirectory + "memory\\";

            // Формирование полного пути к файлу весов
            pathFileWeights = pathDirWeights + name_Layer + "_memory.csv";

            lastdeltaweights = new double[non, nopn + 1];

            double[,] Weights; // Временный массив синаптических весов текущего слоя

            if (File.Exists(pathFileWeights))
                Weights = WeightsInitialize(MemoryMode.GET, pathFileWeights);
            else
            {
                Directory.CreateDirectory(pathDirWeights);
                Weights = WeightsInitialize(MemoryMode.INIT, pathFileWeights);
            }

            for (int i = 0; i < non; i++)
            {
                double[] tmp_weights = new double[nopn + 1];
                for (int j = 0; j < nopn + 1; j++)
                {
                    tmp_weights[j] = Weights[i, j];
                }
                Neurons[i] = new Neuron(tmp_weights, nt); // Заполнение массива нейронами
            }
        }

        // Направляем массив сюда
        public double[,] WeightsInitialize(MemoryMode mm, string path)
        {
            Random random = new Random();

            char[] delim = new char[] { ';', ' ' };
            string[] tmpStrWeights;
            double[,] weights = new double[numofneurons, numofprevneurons + 1];

            // Выбор режима работы с памятью
            switch (mm)
            {
                // Режим загрузки весов
                case MemoryMode.GET:
                    tmpStrWeights = File.ReadAllLines(path);
                    string[] memory_element;

                    for (int i = 0; i < numofneurons; i++)
                    {
                        memory_element = tmpStrWeights[i].Split(delim);
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            weights[i, j] = double.Parse(memory_element[j].Replace(',', '.'),
                                System.Globalization.CultureInfo.InvariantCulture);
                        }
                    }
                    break;

                // Режим сохранения весов в файл
                case MemoryMode.SET:
                    string strSET = "";
                    for (int i = 0; i < numofneurons; i++)
                    {
                        strSET += Neurons[i].Weights[0].ToString(System.Globalization.CultureInfo.InvariantCulture);
                        for (int j = 1; j < numofprevneurons + 1; j++)
                        {
                            strSET += ";" + Neurons[i].Weights[j].ToString(System.Globalization.CultureInfo.InvariantCulture);
                        }
                        if (i < numofneurons - 1)
                            strSET += "\n";
                    }
                    File.WriteAllText(path, strSET);
                    break;

                // Режим инициализации случайными весами
                case MemoryMode.INIT:
                    weights = RandomInit(numofneurons, numofprevneurons + 1);
                    weights = SrChanger(numofneurons, numofprevneurons + 1, weights);
                    break;


            }
            return weights;
        }

        // Случайная инициализация весов в диапазоне [-1, +1]
        protected double[,] RandomInit(int a, int b)
        {
            double[,] weights = new double[a, b];
            Random random = new Random();

            for (int i = 0; i < a; i++)
            {
                for (int j = 0; j < b; j++)
                {
                    weights[i, j] = random.NextDouble() * 2 - 1;
                }
            }
            return weights;
        }

        // Нормализация весов: создаёт отклонение = 1 и мат. ожидание = 0
        // Согласно лекции: "Матожидание = 0, Стандартное отклонение = 1"
        protected double[,] SrChanger(int a, int b, double[,] cweights)
        {
            double[,] weights = cweights;

            for (int i = 0; i < a; i++)
            {
                // Вычисляем среднее (матожидание)
                double mean = 0;
                for (int j = 0; j < b; j++)
                {
                    mean += weights[i, j];
                }
                mean /= b;

                // Центрируем (матожидание = 0)
                for (int j = 0; j < b; j++)
                {
                    weights[i, j] -= mean;
                }

                // Вычисляем стандартное отклонение
                double stdDev = 0;
                for (int j = 0; j < b; j++)
                {
                    stdDev += weights[i, j] * weights[i, j];
                }
                stdDev = Math.Sqrt(stdDev / b);

                // Нормализуем (stdDev = 1)
                if (stdDev > 0)
                {
                    for (int j = 0; j < b; j++)
                    {
                        weights[i, j] /= stdDev;
                    }
                }
            }
            return weights;
        }

        abstract public void Recognize(Network net, Layer nextLayer); // Для прямых проходов
        abstract public double[] BackwardPass(double[] stuff); // Для обратных проходов
    }
}
