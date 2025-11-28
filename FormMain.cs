using System;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using System.IO;
using MO_32_2_Chuprova_BoxBox.NeuroNet;

namespace MO_32_2_Chuprova_BoxBox
{
    public partial class FormMain : Form
    {
        private double[] inputPixels; // Массив входных данных
        private Network network; // Объявление нейросети

        // Конструктор
        public FormMain()
        {
            InitializeComponent();

            inputPixels = new double[15];
            network = new Network();
        }

        // Обработчик события клика кнопки-пикселя
        private void Changing_State_Pixel_Button_Click(object sender, EventArgs e)
        {
            // Если изначально кнопка определенного цвета
            if (((Button)sender).BackColor == Color.LightPink)
            {
                ((Button)sender).BackColor = Color.DarkTurquoise; // Изменение цвета кнопки
                inputPixels[((Button)sender).TabIndex] = 1d; // Изменение в массиве
            }
            // Если у кнопки уже изменённый цвет
            else
            {
                ((Button)sender).BackColor = Color.LightPink; // Изменение цвета кнопки
                inputPixels[((Button)sender).TabIndex] = 0d; // Изменение в массиве
            }
        }

        // Сохранение в файл ОБУЧАЮЩЕГО примера
        private void button_SaveTrainSample_Click(object sender, EventArgs e)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory + "train.txt";
            string tmpStr = numericUpDown_Output.Value.ToString();

            for (int i = 0; i < inputPixels.Length; i++)
            {
                tmpStr += " " + inputPixels[i].ToString();
            }
            tmpStr += "\n";

            File.AppendAllText(path, tmpStr); // Добавление текста tmpStr в файл, расположенный по path
        }

        // Сохранение в файл ТЕСТОВОГО примера
        private void button_SaveTestSample_Click(object sender, EventArgs e)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory + "test.txt";
            string tmpStr = numericUpDown_Output.Value.ToString();

            for (int i = 0; i < inputPixels.Length; i++)
            {
                tmpStr += " " + inputPixels[i].ToString();
            }
            tmpStr += "\n";

            File.AppendAllText(path, tmpStr); // Добавление текста tmpStr в файл, расположенный по path
        }




        // Обработчик события клика кнопки "Обучить"
        private void button16_Click(object sender, EventArgs e)
        {
            network.Train(network);

            // Добавляем данные ошибки на первый график (chart_Earn)
            for (int i = 0; i < network.E_error_avr.Length; i++)
            {
                chart_Earn.Series[0].Points.AddY(network.E_error_avr[i]);
            }

            // Добавляем данные точности на второй график (chart_Accuracy)
            for (int i = 0; i < network.Train_accuracy.Length; i++)
            {
                chart_Accuracy.Series[0].Points.AddY(network.Train_accuracy[i]);
            }

            // MessageBox.Show("Обучение успешно завершено", "Информация", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        // Обработчик события клика кнопки "Распознать"
        private void buttonRecognize_Click(object sender, EventArgs e)
        {
            network.ForwardPass(network, inputPixels);
            labelOut.Text = network.Fact.ToList().IndexOf(network.Fact.Max()).ToString();
            labelRecognize.Text = (100 * network.Fact.Max()).ToString("0.00") + "%";
        }

        // Обработчик события клика кнопки "Тестировать"
        private void buttonTest_Click(object sender, EventArgs e)
        {
            network.Test(network);

            for (int i = 0; i < network.E_error_avr.Length; i++)
            {
                chart_Earn.Series[0].Points.AddY(network.E_error_avr[i]);
            }

            //for (int i = 0; i < network.Train_accuracy.Length; i++)
            //{
            //    chart_Accuracy.Series[1].Points.AddY(network.Train_accuracy[i]);
            //}

            // MessageBox.Show("Тестирование успешно завершено", "Информация", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }
    }
}
