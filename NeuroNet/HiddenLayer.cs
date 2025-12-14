namespace MO_32_2_Chuprova_BoxBox.NeuroNet
{
    class HiddenLayer : Layer
    {
        public HiddenLayer(int non, int nopn, NeuronType nt, string nm_layer, ActivationType actType = ActivationType.Tanh, double dropoutProb = 0.0) : base(non, nopn, nt, nm_layer, actType, dropoutProb) { }

        // Прямой проход
        //public override void Recognize(Network net, Layer nextLayer, bool isTraining = false)
        //{
        //    double[] hidden_out = new double[numofneurons];
        //    for (int i = 0; i < numofneurons; i++)
        //    {
        //        hidden_out[i] = neurons[i].Output;

        //        // Масштабирование выхода при Dropout во время обучения
        //        if (isTraining && dropoutProbability > 0 && !neurons[i].IsDropped)
        //        {
        //            hidden_out[i] /= (1 - dropoutProbability);
        //        }

        //    }
        //    nextLayer.SetData(hidden_out, isTraining);
        //}

        public override void Recognize(Network net, Layer nextLayer, bool isTraining = false)
        {
            double[] hidden_out = new double[numofneurons];
            int activeNeurons = 0;

            for (int i = 0; i < numofneurons; i++)
            {
                hidden_out[i] = neurons[i].Output;
                if (!neurons[i].IsDropped)
                {
                    activeNeurons++;
                }
            }

            // Масштабирование только если есть отключенные нейроны
            if (isTraining && dropoutProbability > 0 && activeNeurons < numofneurons)
            {
                double scale = 1.0 / (1.0 - dropoutProbability);
                for (int i = 0; i < numofneurons; i++)
                {
                    if (!neurons[i].IsDropped)
                    {
                        hidden_out[i] *= scale;
                    }
                }
            }

            nextLayer.SetData(hidden_out, isTraining);
        }

        // Обратный проход (без изменений, кроме использования SetData)
        public override double[] BackwardPass(double[] gr_sums)
        {
            // Градиенты для предыдущего слоя
            double[] gr_sum = new double[numofprevneurons];

            // ВЫЧИСЛЕНИЕ ДЕЛЬТЫ ДЛЯ СКРЫТОГО СЛОЯ
            double[] deltas = new double[numofneurons];
            for (int i = 0; i < numofneurons; i++)
            {
                // Пропускаем отключенные нейроны при Dropout
                if (neurons[i].IsDropped)
                {
                    deltas[i] = 0;
                    continue;
                }

                deltas[i] = gr_sums[i] * neurons[i].Derivative;
            }

            // Вычисление градиентных сумм для предыдущего слоя
            for (int j = 0; j < numofprevneurons; j++)
            {
                double sum = 0;
                for (int k = 0; k < numofneurons; k++)
                {
                    // Пропускаем отключённые нейроны
                    if (!neurons[k].IsDropped)
                    {
                        sum += neurons[k].Weights[j + 1] * deltas[k];
                    }
                }
                gr_sum[j] = sum;
            }

            // КОРРЕКЦИЯ ВЕСОВ СКРЫТОГО СЛЯ
            for (int i = 0; i < numofneurons; i++)
            {
                // Не обновляем веса отключенных нейронов
                if (neurons[i].IsDropped)
                    continue;

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