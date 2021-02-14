TAXI_ENVIROMENT_REWARDS = dict(
    step=-1,
    no_fuel=-20,
    bad_pickup=-15,
    bad_dropoff=-15,
    bad_refuel=-10,
    bad_fuel=-50,
    pickup=50,
    standby_engine_off=-1,
    turn_engine_on=-10e6,
    turn_engine_off=-10e6,
    standby_engine_on=-1,
    intermediate_dropoff=50,
    final_dropoff=100,
    hit_wall=-2,
    collision=-35,
    collided=-20,
    unrelated_action=-15,
)

NEW_TAXI_ENVIRONMENT_REWARDS = dict(
    step=-1,
    no_fuel=-20,
    bad_pickup=-15,
    bad_dropoff=-15,
    bad_refuel=-10,
    bad_fuel=-50,
    pickup=50,
    standby_engine_off=-1,
    turn_engine_on=-10e6,
    turn_engine_off=-10e6,
    standby_engine_on=-1,
    intermediate_dropoff=50,
    final_dropoff=100,
    hit_wall=-2,
    collision=-35,
    collided=-20,
    unrelated_action=-15,
)

ALL_ACTIONS_NAMES = ['south', 'north', 'east', 'west',
                     'pickup', 'dropoff', 'refuel']
# ALL_ACTIONS_NAMES = ['south', 'north', 'east', 'west',
#                      'pickup', 'dropoff', 'bind',
#                      'turn_engine_on', 'turn_engine_off',
#                      'standby',
#                      'refuel']

BASE_AVAILABLE_ACTIONS = ['south', 'north', 'east', 'west',
                          'pickup', 'dropoff']
