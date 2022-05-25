# def find_top_20(candidates):
#     result = []  # список тюплов - (суммы баллов, оценка информатики, оценка математики, имя)
#     for enrollee in candidates:
#         score = enrollee["scores"]["math"] + enrollee["scores"]["russian_language"] + enrollee["scores"][
#             "computer_science"] + enrollee["extra_scores"]  # сумма баллов
#         result.append((score, enrollee["scores"]["computer_science"], enrollee["scores"]["math"], enrollee["name"]))
#     result.sort(reverse=True)
#     scores = [data_set[0] for data_set in result]  # список из суммы баллов
#     start_index = scores.index(result[19][0])  # начальный индекс абитуирента с суммой баллов 20 позиции
#     stop_index = start_index + scores.count(result[19][0])  # конечный индекс абитуирента с суммой баллов 20 позиции
#     if start_index != stop_index - 1:
#         temp = [data[1:] for data in result[start_index:stop_index]]
#         temp.sort(reverse=True)
#         scores_inf = [data_set[0] for data_set in temp]  # список из оценок по информатике
#         start_i = scores_inf.index(temp[19 - start_index][0])  # начальный с одинаковой оценкой  по информатике
#         stop_i = start_i + scores_inf.count(temp[19 - start_index][0])  # конечный с одинаковой оценкой  по информатике
#         result = [data[3] for data in result[:start_index]]  # кандидаты с разными суммами баллов
#         # добавляем кандидатов с одинаковой суммой баллов, но разной оценки по информатике
#         result.extend(data[2] for data in temp[:len(temp) - scores_inf.count(temp[19 - start_index][0])])
#         if start_i != stop_i - 1:
#             temp = [data[1:] for data in temp[start_i:stop_i]]  # список из списков оценок по математике и имени
#             temp = [data[1] for data in sorted(temp, reverse=True)]  # список имен. сортировка по оценке по математике
#             result.extend(temp)
#         result = result[:20]
#     else:
#         result = [data[3] for data in result[:20]]
#
#     return result
#
#
# candidates = [
#     {"name": "Vasya", "scores": {"math": 58, "russian_language": 62, "computer_science": 48}, "extra_scores": 0},  # 168
#     {"name": "Fedya", "scores": {"math": 34, "russian_language": 85, "computer_science": 43}, "extra_scores": 2},  # 162
#     {"name": "Petya", "scores": {"math": 92, "russian_language": 33, "computer_science": 34}, "extra_scores": 1},  # 160
#     {"name": "Vasya2", "scores": {"math": 58, "russian_language": 62, "computer_science": 48}, "extra_scores": 1},  # 168
#     {"name": "Fedya2", "scores": {"math": 33, "russian_language": 85, "computer_science": 43}, "extra_scores": 3},  # 162
#     {"name": "Petya2", "scores": {"math": 92, "russian_language": 33, "computer_science": 34}, "extra_scores": 2},  # 160
#     {"name": "Vasya3", "scores": {"math": 58, "russian_language": 62, "computer_science": 48}, "extra_scores": 2},  # 168
#     {"name": "Fedya3", "scores": {"math": 33, "russian_language": 85, "computer_science": 42}, "extra_scores": 3},  # 162
#     {"name": "Petya3", "scores": {"math": 92, "russian_language": 33, "computer_science": 34}, "extra_scores": 3},  # 160
#     {"name": "Vasya4", "scores": {"math": 58, "russian_language": 62, "computer_science": 48}, "extra_scores": 3},  # 168
#     {"name": "Fedya4", "scores": {"math": 33, "russian_language": 85, "computer_science": 42}, "extra_scores": 5},  # 162
#     {"name": "Petya4", "scores": {"math": 92, "russian_language": 33, "computer_science": 35}, "extra_scores": 3},  # 160
#     {"name": "Vasya5", "scores": {"math": 58, "russian_language": 62, "computer_science": 48}, "extra_scores": 4},  # 168
#     {"name": "Fedya5", "scores": {"math": 33, "russian_language": 85, "computer_science": 42}, "extra_scores": 6},  # 162
#     {"name": "Petya5", "scores": {"math": 91, "russian_language": 33, "computer_science": 35}, "extra_scores": 4},  # 160
#     {"name": "Vasya6", "scores": {"math": 58, "russian_language": 62, "computer_science": 48}, "extra_scores": 5},  # 168
#     {"name": "Fedya6", "scores": {"math": 33, "russian_language": 85, "computer_science": 42}, "extra_scores": 7},  # 162
#     {"name": "Petya6", "scores": {"math": 92, "russian_language": 33, "computer_science": 34}, "extra_scores": 6},  # 160
#     {"name": "Vasya7", "scores": {"math": 58, "russian_language": 62, "computer_science": 48}, "extra_scores": 6},  # 168
#     {"name": "Fedya7", "scores": {"math": 33, "russian_language": 85, "computer_science": 42}, "extra_scores": 8},  # 162
#     {"name": "Petya7", "scores": {"math": 92, "russian_language": 33, "computer_science": 34}, "extra_scores": 7},  # 160
#     {"name": "Vasya8", "scores": {"math": 58, "russian_language": 62, "computer_science": 48}, "extra_scores": 7},  # 168
#     {"name": "Fedya8", "scores": {"math": 33, "russian_language": 85, "computer_science": 42}, "extra_scores": 9},  # 162
#     {"name": "Petya8", "scores": {"math": 92, "russian_language": 33, "computer_science": 34}, "extra_scores": 8}  # 160
# ]
#
# print(find_top_20(candidates))

# def get_inductees(names, birthday_years, genders):
#     subject_to_conscription = []
#     unable_to_install = []
#     for gender, birthday_year, name in zip(genders, birthday_years, names):
#         if gender == None or gender == "Male" and birthday_year == None:
#             unable_to_install.append(name)
#         elif gender == "Male" and 18 <= 2021 - birthday_year < 30:
#             subject_to_conscription.append(name)
#
#     return subject_to_conscription, unable_to_install
#
# names = ["Vasya","Alice","Petya","Jenny","Fedya","Viola","Mark","Chris","Margo"]
# birthday_years = [1962,1995,2000,None,None,None,None,1998,2001]
# genders = ["Male","Female","Male","Female","Male",None,None,None,None]
#
# print(get_inductees(names, birthday_years, genders))

# def find_athlets(know_english, sportsmen, more_than_20_years):
#     result = set(know_english) & set(sportsmen) & set(more_than_20_years)
#
#     return list(result)
#
# know_english = ["Vasya", "Jimmy", "Max", "Peter", "Eric", "Zoi", "Felix"]
# sportsmen = ["Don", "Peter", "Eric", "Jimmy", "Mark"]
# more_than_20_years = ["Peter", "Julie", "Jimmy", "Mark", "Max"]
#
# print(find_athlets(know_english, sportsmen, more_than_20_years))

import openpyxl


def make_report_about_top3(students_avg_scores):
    res = sorted(((k, v) for d in [students_avg_scores] for k, v in d.items()), key=lambda pair: pair[1], reverse=True)
    wb = openpyxl.Workbook()
    list = wb.active
    list.append(('Name', 'Avg score'))  # # Создание строки с заголовками
    for name in res[:3]:
        list.append(name)
    wb.save('names_avg_scores.xlsx')
    return 'names_avg_scores.xlsx'


students_avg_scores = {'Max': 4.964, 'Eric': 4.962, 'Peter': 4.923, 'Mark': 4.957, 'Julie': 4.95, 'Jimmy': 4.973,
                       'Felix': 4.937, 'Vasya': 4.911, 'Don': 4.936, 'Zoi': 4.937}

print(make_report_about_top3(students_avg_scores))
