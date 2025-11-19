using System;
using System.Linq;

namespace MO_32_2_Chuprova_BoxBox.NeuroNet
{
    class OutputLayer : Layer
    {
        public OutputLayer(int non, int nopn, NeuronType nt, string nm_layer) : base(non, nopn, nt, nm_layer) { }

        // Прямой проход 
        // Softmax преобразует выходы в вероятности (сумма выходов = 1)
        public override void Recognize(Network net, Layer nextLayer)
        {
            double max = neurons.Max(n => n.Output);
            double sum = 0;

            double[] expVals = new double[neurons.Length];

            for (int i = 0; i < neurons.Length; i++)
            {
                expVals[i] = Math.Exp(neurons[i].Output - max);
                sum += expVals[i];
            }

            for (int i = 0; i < neurons.Length; i++)
                net.Fact[i] = expVals[i] / sum;
        }

        // Обратный проход
        public override double[] BackwardPass(double[] errors)
        {
            // Размерность должна соответствовать количеству нейронов в предыдущем слое
            double[] gr_sum = new double[numofprevneurons + 1];

            // Вычисление градиентных сумм выходного слоя
            for (int j = 0; j < numofprevneurons; j++)
            {
                double sum = 0;
                for (int k = 0; k < numofneurons; k++)
                    sum += neurons[k].Weights[j] * errors[k];

                gr_sum[j] = sum;
            }

            // Цикл коррекции синаптических весов
            for (int i = 0; i < numofneurons; i++)
            {
                for (int n = 0; n < numofprevneurons + 1; n++)
                {
                    double deltaw;
                    if (n == 0) // Коррекция веса порога
                    {
                        // Для порога используем только ошибку и момент
                        deltaw = momentum * lastdeltaweights[i, 0] + learningrate * errors[i];
                    }
                    else // Коррекция обычных весов
                    {
                        // Умножаем ошибку на соответствующий вход
                        deltaw = momentum * lastdeltaweights[i, n] + learningrate * neurons[i].Inputs[n - 1] * errors[i];
                    }

                    lastdeltaweights[i, n] = deltaw;
                    neurons[i].Weights[n] += deltaw; // Коррекция весов
                }
            }

            return gr_sum;
        }
    }
}
