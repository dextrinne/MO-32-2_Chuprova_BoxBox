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

    // Новое перечисление для типов активации
    enum ActivationType // Тип функции активации
    {
        Sigmoid,     // Сигмоидальная (логистическая)
        Tanh,        // Гиперболический тангенс
        ReLU,        // Выпрямленная линейная единица
        LeakyReLU,   // Утечка ReLU
        UnitStep,     // Единичный скачок
        Linear
    }
}
