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
    {"name": "Fedya", "scores": {"math": 33, "russian_language": 85, "computer_science": 43}, "extra_scores": 2},  # 162
    {"name": "Petya", "scores": {"math": 92, "russian_language": 33, "computer_science": 34}, "extra_scores": 1},  # 160
    {"name": "Vasya2", "scores": {"math": 58, "russian_language": 62, "computer_science": 48}, "extra_scores": 1},  # 168
    {"name": "Fedya2", "scores": {"math": 33, "russian_language": 85, "computer_science": 43}, "extra_scores": 3},  # 162
    {"name": "Petya2", "scores": {"math": 92, "russian_language": 33, "computer_science": 34}, "extra_scores": 2},  # 160
    {"name": "Vasya3", "scores": {"math": 58, "russian_language": 62, "computer_science": 48}, "extra_scores": 2},  # 168
    {"name": "Fedya3", "scores": {"math": 33, "russian_language": 85, "computer_science": 42}, "extra_scores": 4},  # 162
    {"name": "Petya3", "scores": {"math": 92, "russian_language": 33, "computer_science": 34}, "extra_scores": 3},  # 160
    {"name": "Vasya4", "scores": {"math": 58, "russian_language": 62, "computer_science": 48}, "extra_scores": 3},  # 168
    {"name": "Fedya4", "scores": {"math": 33, "russian_language": 85, "computer_science": 42}, "extra_scores": 5},  # 162
    {"name": "Petya4", "scores": {"math": 92, "russian_language": 33, "computer_science": 34}, "extra_scores": 4},  # 160
    {"name": "Vasya5", "scores": {"math": 58, "russian_language": 62, "computer_science": 48}, "extra_scores": 4},  # 168
    {"name": "Fedya5", "scores": {"math": 33, "russian_language": 85, "computer_science": 42}, "extra_scores": 6},  # 162
    {"name": "Petya5", "scores": {"math": 91, "russian_language": 33, "computer_science": 34}, "extra_scores": 5},  # 160
    {"name": "Vasya6", "scores": {"math": 58, "russian_language": 62, "computer_science": 48}, "extra_scores": 5},  # 168
    {"name": "Fedya6", "scores": {"math": 33, "russian_language": 85, "computer_science": 42}, "extra_scores": 7},  # 162
    {"name": "Petya6", "scores": {"math": 92, "russian_language": 33, "computer_science": 34}, "extra_scores": 6},  # 160
    {"name": "Vasya7", "scores": {"math": 58, "russian_language": 62, "computer_science": 48}, "extra_scores": 6},  # 168
    {"name": "Fedya7", "scores": {"math": 33, "russian_language": 85, "computer_science": 42}, "extra_scores": 8},  # 162
    {"name": "Petya7", "scores": {"math": 92, "russian_language": 33, "computer_science": 34}, "extra_scores": 7},  # 160
    {"name": "Vasya8", "scores": {"math": 58, "russian_language": 62, "computer_science": 48}, "extra_scores": 7},  # 168
    {"name": "Fedya8", "scores": {"math": 33, "russian_language": 85, "computer_science": 42}, "extra_scores": 9},  # 162
    {"name": "Petya8", "scores": {"math": 92, "russian_language": 33, "computer_science": 34}, "extra_scores": 8}  # 160
]

print(find_top_20(candidates))
