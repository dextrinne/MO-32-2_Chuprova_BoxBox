namespace MO_32_2_Chuprova_BoxBox.NeuroNet
{
    class HiddenLayer : Layer
    {
        public HiddenLayer(int non, int nopn, NeuronType nt, string nm_layer):base(non, nopn, nt, nm_layer) { }

        // Прямой проход 
        public override void Recognize(Network net, Layer nextLayer)
        {
            double[] hidden_out = new double[numofneurons];
            for (int i = 0; i < numofneurons; i++)
            {
                hidden_out[i] = neurons[i].Output;
            }
            nextLayer.Data = hidden_out; // Передача выходного сигнала на вход следующего слоя
        }

        // Обратный проход
        public override double[] BackwardPass(double[] gr_sums)
        {
            // Градиенты для предыдущего слоя
            double[] gr_sum = new double[numofprevneurons];

            // ВЫЧИСЛЕНИЕ ДЕЛЬТЫ ДЛЯ СКРЫТОГО СЛОЯ
            double[] deltas = new double[numofneurons];
            for (int i = 0; i < numofneurons; i++)
            {
                deltas[i] = gr_sums[i] * neurons[i].Derivative;
            }

            // Вычисление градиентных сумм для предыдущего слоя
            for (int j = 0; j < numofprevneurons; j++)
            {
                double sum = 0;
                for (int k = 0; k < numofneurons; k++)
                {
                    sum += neurons[k].Weights[j + 1] * deltas[k];
                }
                gr_sum[j] = sum;
            }

            // КОРРЕКЦИЯ ВЕСОВ СКРЫТОГО СЛОЯ
            for (int i = 0; i < numofneurons; i++)
            {
                for (int n = 0; n < numofprevneurons + 1; n++)
                {
                    double deltaw;
                    if (n == 0) // Вес порога
                    {
                        deltaw = learningrate * deltas[i] + momentum * lastdeltaweights[i, 0];
                    }
                    else // Обычные веса
                    {
                        deltaw = learningrate * neurons[i].Inputs[n - 1] * deltas[i] + momentum * lastdeltaweights[i, n];
                    }

                    lastdeltaweights[i, n] = deltaw;
                    neurons[i].Weights[n] += deltaw;
                }
            }

            return gr_sum;
        }
    }
}
