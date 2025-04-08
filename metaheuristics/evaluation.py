def relative_to_greedy(solution: dict, greedy: dict) -> float:

    n=len(greedy)
    value= sum(
        solution[name]/greedy[name] 
        for name in greedy.keys()
        )
    
    return format_value(value/n)

def sum_of_distances(solution: dict) -> float:
    value = sum(solution.values())
    return format_value(value)

def format_value(value):
    #return round(value/ 1_000, 4)
    return value