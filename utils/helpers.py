def separated_instances(instances: list, slice_interval: int=9) -> list:
    size = len(instances)
    interval = 0
    instances_split = []
    while interval < size:
        instance = instances[interval:interval+slice_interval]
        instances_split.append(instance)
        interval += slice_interval
    return instances_split

def get_instances(instances: list, slice_interval: list=[6, 15, 30, 45]) -> list:
    instances_split = []
    for s_i in slice_interval:
        instances_split.extend(separated_instances(instances, s_i))
    return instances_split

