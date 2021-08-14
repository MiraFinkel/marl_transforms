import os


def read_pddl(f_name):
    data = None
    with open(f_name) as f:
        data = f.readlines()
    return data


def save_pddl_file(data, f_name):
    with open(f_name, 'w') as f:
        f.write("".join(data))


def get_pddl_pred(data):
    init_pred = "precondition"
    end_pred = ")"

    relevent_scope = False
    all_pre = set()
    for line in data:
        if init_pred in line:
            relevent_scope = True
            continue
        elif end_pred == line.strip():
            relevent_scope = False
            continue
        if relevent_scope:
            all_pre.add(line.strip())
    return list(all_pre)


def relax_preds(pddl_data, pre_list):
    new_pddl = []
    init_pred = "precondition"
    end_pred = ")"
    relevent_scope = False
    for line in pddl_data:
        if init_pred in line:  # DETECT WE ARE IN THE RIGHT SCOPE
            relevent_scope = True
            new_pddl.append(line)
            continue
        elif end_pred == line.strip():  # DETECT WE ARE OUT OF THE RIGHT SCOPE
            relevent_scope = False
            new_pddl.append(line)
            continue
        if relevent_scope:  # ONLY IF WE ARE LOOKING INTO PRECONDITIONS
            pre_in_line = False
            for pre in pre_list:
                if pre in line:
                    pre_in_line = True
                    break
            if pre_in_line:
                # print("line skipped", line)
                continue
        new_pddl.append(line)
    return new_pddl


if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    pddl_example = read_pddl(curr_dir + "/" + "example.pddl")  # reads or example file
    all_pres = get_pddl_pred(pddl_example)
    some_preds = all_pres[:3]
    print("to remove pre list", some_preds)
    new_pddl = relax_preds(pddl_example, some_preds)
    save_pddl_file(new_pddl, curr_dir + "/new.pddl")
