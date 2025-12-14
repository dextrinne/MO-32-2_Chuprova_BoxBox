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
        //public FormMain()
        //{
        //    InitializeComponent();

        //    inputPixels = new double[15];
        //    network = new Network();
        //}
        public FormMain()
        {
            InitializeComponent();

            // Инициализация элементов управления
            InitializeCustomControls();

            inputPixels = new double[15];
            network = new Network(); // Начальная инициализация
        }

        // Функция инициализации пользовательских элементов
        private void InitializeCustomControls()
        {
            // Устанавливаем начальные значения
            if (comboBoxActivation != null)
            {
                comboBoxActivation.SelectedIndex = 1; // Гиперболический тангенс по умолчанию
            }

            if (trackBarDropout != null)
            {
                trackBarDropout.Value = 0; // 0% по умолчанию
                labelDropoutValue.Text = "0%";
            }
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

        // Обработчик изменения значения Dropout
        private void TrackBarDropout_ValueChanged(object sender, EventArgs e)
        {
            // Обновляем Label с текущим значением
            if (labelDropoutValue != null)
            {
                labelDropoutValue.Text = $"{trackBarDropout.Value}%";
            }
        }

        // Обработчик нажатия кнопки применения настроек
        private void ButtonApplySettings_Click(object sender, EventArgs e)
        {
            try
            {
                // Проверяем, что ComboBox что-то выбрал
                if (comboBoxActivation.SelectedItem == null)
                {
                    MessageBox.Show("Выберите функцию активации!", "Ошибка",
                                  MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }

                // Преобразуем выбранный текст в тип ActivationType
                ActivationType activationType = ActivationType.Tanh; // Значение по умолчанию

                switch (comboBoxActivation.SelectedItem.ToString())
                {
                    case "Сигмоидальная":
                        activationType = ActivationType.Sigmoid;
                        break;
                    case "Гиперболический тангенс":
                        activationType = ActivationType.Tanh;
                        break;
                    case "ReLU":
                        activationType = ActivationType.ReLU;
                        break;
                    case "Leaky ReLU":
                        activationType = ActivationType.LeakyReLU;
                        break;
                    case "Единичный скачок":
                        activationType = ActivationType.UnitStep;
                        break;
                    default:
                        MessageBox.Show("Неизвестный тип активации!", "Ошибка",
                                      MessageBoxButtons.OK, MessageBoxIcon.Error);
                        return;
                }

                // Получаем Dropout rate (преобразуем проценты в дробь)
                double dropoutRate = trackBarDropout.Value / 100.0;

                // Проверяем допустимый диапазон
                if (dropoutRate < 0 || dropoutRate > 0.5)
                {
                    MessageBox.Show("Dropout должен быть от 0% до 50%!", "Ошибка",
                                  MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }

                // Создаем новую сеть с выбранными параметрами
                network = new Network(activationType, dropoutRate);

                // Показываем сообщение об успехе
                MessageBox.Show($"Настройки успешно применены:\n\n" +
                               $"Функция активации: {comboBoxActivation.SelectedItem}\n" +
                               $"Dropout rate: {dropoutRate:P0} ({trackBarDropout.Value}%)\n\n" +
                               $"Нейросеть будет использовать эти настройки при следующем обучении.",
                               "Настройки применены",
                               MessageBoxButtons.OK,
                               MessageBoxIcon.Information);

                // Очищаем графики (опционально)
                ClearCharts();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Ошибка при применении настроек:\n{ex.Message}",
                               "Ошибка",
                               MessageBoxButtons.OK,
                               MessageBoxIcon.Error);
            }
        }

        // Функция для очистки графиков
        private void ClearCharts()
        {
            // Очищаем график ошибок
            if (chart_Earn != null && chart_Earn.Series.Count > 0)
            {
                chart_Earn.Series[0].Points.Clear();
            }

            // Очищаем график точности
            if (chart_Accuracy != null && chart_Accuracy.Series.Count > 0)
            {
                chart_Accuracy.Series[0].Points.Clear();
            }

            // Сбрасываем метки результатов
            labelOut.Text = "Out";
            labelRecognize.Text = "Вероятность";
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
