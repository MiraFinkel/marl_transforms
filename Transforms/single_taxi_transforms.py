from Environments.SingleTaxiEnv.single_taxi_wrapper import *
from Transforms.transform_constants import *


class SingleTaxiTransformedEnv(SingleTaxiSimpleEnv):
    def __init__(self, transforms):
        super().__init__(False)
        self.taxi_x_transform, self.taxi_y_transform = transforms[0], transforms[1]
        self.pass_loc_transform, self.pass_dest_transform = transforms[2], transforms[3]
        self.fuel_transform = transforms[4]
        self.all_outcome_determinization = transforms[5]
        self.most_likely_outcome = transforms[6]
        if self.all_outcome_determinization:
            self.P = p_determinization(self.P)


    def step(self, a):
        s, r, d, p = super(SingleTaxiTransformedEnv, self).step(a)
        next_state = self.decode(s)
        taxi_x, taxi_y, pass_loc_idx, pass_dest_idx, fuel = next_state
        if self.taxi_x_transform:
            taxi_x = 0
        if self.taxi_y_transform:
            taxi_y = 0
        if self.pass_loc_transform:
            pass_loc_idx = 0
        if self.pass_dest_transform:
            pass_dest_idx = 0
        if self.fuel_transform:
            fuel = MAX_FUEL

        transformed_next_state = self.encode(taxi_x, taxi_y, pass_loc_idx, pass_dest_idx, fuel)
        return int(transformed_next_state), r, d, {"prob": p}


def p_determinization(p):
    for (s, s_probs) in p.items():
        for (a, a_probs) in s_probs.items():
            probs_list = [prob[0] for prob in a_probs]
            max_prob = max(probs_list)
            max_prob_idx = probs_list.index(max_prob)
            p[s][a] = tuple([1.0] + list(p[s][a][max_prob_idx])[1:])
    return p


def get_single_taxi_transform_name(transforms):
    taxi_x_transform, taxi_y_transform = transforms[0], transforms[1]
    pass_loc_transform, pass_dest_transform = transforms[2], transforms[3]
    fuel_transform = transforms[4]
    all_outcome_determinization = transforms[5]
    most_likely_outcome = transforms[6]

    name = ""
    name += TAXI_LOC_X if taxi_x_transform else ""
    name += TAXI_LOC_Y if taxi_y_transform else ""
    name += PASS_LOC if pass_loc_transform else ""
    name += PASS_DEST if pass_dest_transform else ""
    name += FUEL if fuel_transform else ""
    name += ALL_OUTCOME_DETERMINIZATION if all_outcome_determinization else ""
    name += MOST_LIKELY_OUTCOME if most_likely_outcome else ""
    return name


if __name__ == '__main__':
    cur_transforms = [False, False, False, False, False, True, False]
    new_env = SingleTaxiTransformedEnv(cur_transforms)
    for _ in range(100):
        next_s, r, d, prob = new_env.step(np.random.randint(0, 6))
        print(new_env.decode(next_s))

    passenger_locations, fuel_station = new_env.get_info_from_map()
    a = 7
