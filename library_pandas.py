# Библиотека Pandas используется для анализа данных в форме таблиц или, как еще говорят, датафреймов.
import pandas as pd

# Создадим таблицу или датафрейм
bank_db = {
    'Name': ['Иван Иванович', 'Иван Петрович'],
    'Age': [0, 1],
    'Experience': [4, 7],
    'Salary': [75, 95],
    'Credit_score': [4, 8],
    'Outcome': [0, 1]
}

df = pd.DataFrame(bank_db)
print(df)

# посчитаем средний доход заемщиков: (75 + 95) / 2
print(df['Salary'].mean())