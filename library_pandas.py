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

def find_top_20(candidates):
    result = []  # список тюплов - (суммы баллов, оценка информатики, математики, имени)
    for enrollee in candidates:
        score = enrollee["scores"]["math"] + enrollee["scores"]["russian_language"] + enrollee["scores"][
            "computer_science"] + enrollee["extra_scores"]
        result.append((score, enrollee["scores"]["computer_science"], enrollee["scores"]["math"], enrollee["name"]))
    result.sort(reverse=True)
    scores = [data_set[0] for data_set in result]  # список из суммы баллов
    start_index = scores.index(result[19][0])  # начальный индекс абитуирента с суммой баллов 20 позиции
    stop_index = start_index + scores.count(result[19][0])  # конечный индекс абитуирента с суммой баллов 20 позиции
    print(result)
    print(0, start_index, stop_index, scores.count(result[19][0]))
    if start_index != stop_index - 1:
        temp = [data[1:] for data in result[start_index:stop_index]]
        temp.sort(reverse=True)
        scores_inf = [data_set[0] for data_set in temp]  # список из оценок по информатике
        print(1, temp)
        print(2, scores_inf)
        start_i = scores_inf.index(temp[19 - start_index][0])  # начальный индекс абитуирента с одинаковой оценкой
        stop_i = start_i + scores_inf.count(temp[19 - start_index][0])  # конечный индекс абитуирента с одинаковой оценкой
        print(3, start_i, stop_i, scores_inf.count(temp[19 - start_index][0]))
        result = [data[3] for data in result[:start_index]]
        print(len(result))
        result.extend(data[2] for data in temp[:start_i])
        print(len(result))
        if start_i != stop_i - 1:
            temp = [data[1:] for data in temp[start_i:stop_i]]
            print(4, temp)
            temp = [data[1] for data in sorted(temp, reverse=True)]
            result.extend(temp)
        result = result[:20]
    else:
    result = [data[3] for data in result[:20]]


# print(temp)
print(len(result))
return result

candidates = [
    {"name": "Vasya", "scores": {"math": 58, "russian_language": 62, "computer_science": 48}, "extra_scores": 0},  # 168
    {"name": "Fedya", "scores": {"math": 33, "russian_language": 85, "computer_science": 42}, "extra_scores": 2},  # 162
    {"name": "Petya", "scores": {"math": 92, "russian_language": 33, "computer_science": 34}, "extra_scores": 1}   # 160
]

print(find_top_20(candidates))
