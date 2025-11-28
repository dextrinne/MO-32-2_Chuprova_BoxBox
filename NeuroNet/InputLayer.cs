using System;
using System.IO;

namespace MO_32_2_Chuprova_BoxBox.NeuroNet
{
    class InputLayer
    {
        // Поля
        private double[,] trainset; // 100 изображений в обучающей выборке
        private double[,] testset; // 10 изображений в тестовой выборке

        // Свойства
        public double[,] Trainset { get => trainset; }
        public double[,] Testset { get => testset; }

        // Конструктор
        public InputLayer(NetworkMode nm)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory; // Директория
            string[] tmpArrStr; // Временный массив строк
            string[] tmpStr; // Временный (вспомогательный) массив элементов в строке

            switch (nm)
            {
                case NetworkMode.Train:
                    tmpArrStr = File.ReadAllLines(path + "train.txt"); // Считываем из файла
                    trainset = new double[tmpArrStr.Length, 16]; // Определение массива обучения

                    for (int i = 0; i < tmpArrStr.Length; i++) // Цикл перебора строк обучающей выборки
                    {
                        tmpStr = tmpArrStr[i].Split(' '); // Разбиение i-той строки на массив отдельных элементов

                        for (int j = 0; j < 16; j++) // Цикл заполнения i-той строки обучающей выборки
                        {
                            trainset[i, j] = double.Parse(tmpStr[j]);
                        }
                    }

                    Shuffling_Array_Rows(trainset); // Перетасовка обучающей выборки (метод Фишера-Йетса)
                    break;

                case NetworkMode.Test:
                    tmpArrStr = File.ReadAllLines(path + "test.txt"); // Считываем из файла
                    testset = new double[tmpArrStr.Length, 16]; // Определение массива тестов

                    for (int i = 0; i < tmpArrStr.Length; i++) // Цикл перебора строк тестовой выборки
                    {
                        tmpStr = tmpArrStr[i].Split(' '); // Разбиение i-той строки на массив отдельных элементов

                        for (int j = 0; j < 16; j++) // Цикл заполнения i-той строки тестовой выборки
                        {
                            testset[i, j] = double.Parse(tmpStr[j]);
                        }
                    }

                    Shuffling_Array_Rows(testset); // Перетасовка тестовой выборки (метод Фишера-Йетса)
                    break;
            }
        }

        // Метод Фишера-Йетса для перетасовки строк матрицы
        // Важно для предотвращения запоминания порядка данных сетью
        public void Shuffling_Array_Rows(double[,] arr)
        {
            if (arr == null) return;

            Random rand = new Random();
            int rowCount = arr.GetLength(0);
            int colCount = arr.GetLength(1);
            double[] tempRow = new double[colCount];

            for (int i = rowCount - 1; i > 0; i--)
            {
                int j = rand.Next(i + 1);

                // Обмен строками i и j
                for (int k = 0; k < colCount; k++)
                    tempRow[k] = arr[i, k];

                for (int k = 0; k < colCount; k++)
                    arr[i, k] = arr[j, k];

                for (int k = 0; k < colCount; k++)
                    arr[j, k] = tempRow[k];
            }
        }
    }
}
