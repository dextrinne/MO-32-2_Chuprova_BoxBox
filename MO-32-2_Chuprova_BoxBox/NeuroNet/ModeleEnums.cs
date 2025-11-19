namespace MO_32_2_Chuprova_BoxBox.NeuroNet
{
    enum MemoryMode // Режим работы памяти
    {
        GET, // Считывание памяти
        SET, // Сохранение памяти
        INIT // Инициализация памяти
    }

    enum NeuronType // Тип нейрона
    {
        Hidden, // Скрытный
        Output // Выходной
    }

    enum NetworkMode // Режим работы сети
    {
        Train, // Обучение
        Test, // Проверка
        Demo // Распознавание
    }
}
