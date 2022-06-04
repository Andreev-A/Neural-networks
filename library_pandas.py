# Библиотека Pandas используется для анализа данных в форме таблиц или, как еще говорят, датафреймов.
# import pandas as pd
#
# # Создадим таблицу или датафрейм
# bank_db = {
#     'Name': ['Иван Иванович', 'Иван Петрович'],
#     'Age': [0, 1],
#     'Experience': [4, 7],
#     'Salary': [75, 95],
#     'Credit_score': [4, 8],
#     'Outcome': [0, 1]
# }
#
# df = pd.DataFrame(bank_db)
# print(df)
#
# # посчитаем средний доход заемщиков: (75 + 95) / 2
# print(df['Salary'].mean())
#
# a = 124
# print(f'{a:05d}')
#
# a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# print([a[i:i + 3] for i in range(0, len(a), 3)])

print((lambda a: a**2)(2))
print((lambda x,y,z: x + y + z)(1,2,3))           # Пример позиционной подачи параметров
print((lambda x,y,z=3: x + y + z)(1,2))           # Пример значения по умолчанию
print((lambda x,y,z=3: x + y + z)(1, y=2))        # Пример именованного параметра
print((lambda *arg: arg[0] + arg[1])(1,2))        # Пример подачи набора параметров
print((lambda **kwarg: kwarg.values())(x=1,y=2))  # Пример подачи именованных параметров