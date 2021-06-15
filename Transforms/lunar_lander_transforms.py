from Environments.LunarLander.lunar_lander import *
from Transforms.transform_constants import *


class LunarLanderTransformedEnv(LunarLander):
    def __init__(self, transforms):
        self.in_reset = True
        super().__init__()
        self.in_reset = False

        x_coordinate, y_coordinate = transforms[0], transforms[1]
        x_horizontal_velocity, y_vertical_velocity = transforms[2], transforms[3]
        orientation_in_space = transforms[4]
        angular_velocity = transforms[5]

        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.x_horizontal_velocity = x_horizontal_velocity
        self.y_vertical_velocity = y_vertical_velocity
        self.orientation_in_space = orientation_in_space
        self.angular_velocity = angular_velocity

    def step(self, action):
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[
                1]  # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            # particles are just a decoration, 3.5 is here to make particle speed adequate
            p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)
            p.ApplyLinearImpulse((ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power), impulse_pos,
                                 True)
            self.lander.ApplyLinearImpulse((-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                                           impulse_pos, True)

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1, 3]):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert 0.5 <= s_power <= 1.0
            else:
                direction = action - 2
                s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                           self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse((ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power), impulse_pos,
                                 True)
            self.lander.ApplyLinearImpulse((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                                           impulse_pos, True)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0
        ]
        assert len(state) == 8

        reward = 0
        shaping = \
            - 100 * np.sqrt(state[0] * state[0] + state[1] * state[1]) \
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3]) \
            - 100 * abs(state[4]) + 10 * state[6] + 10 * state[7]  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power * 0.30  # less fuel spent is better, about -30 for heurisic landing
        reward -= s_power * 0.03

        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done = True
            reward = -100
        if not self.lander.awake:
            done = True
            reward = +100
        if not self.in_reset:
            if self.x_coordinate:
                state[0] = 0.0
            if self.y_coordinate:
                state[1] = 0.0
            if self.x_horizontal_velocity:
                state[2] = 0.0
            if self.y_vertical_velocity:
                state[3] = 0.0
            if self.orientation_in_space:
                state[4] = 0.0
            if self.angular_velocity:
                state[5] = 0.0
        return np.array(state, dtype=np.float32), reward, done, {}


def get_lunar_lander_transform_name(transforms):
    x_coordinate, y_coordinate = transforms[0], transforms[1]
    x_horizontal_velocity, y_vertical_velocity = transforms[2], transforms[3]
    orientation_in_space = transforms[4]
    angular_velocity = transforms[5]
    name = ""
    name += X_COORDINATE if x_coordinate else ""
    name += Y_COORDINATE if y_coordinate else ""
    name += X_VELOCITY if x_horizontal_velocity else ""
    name += Y_VELOCITY if y_vertical_velocity else ""
    name += ORIENTATION if orientation_in_space else ""
    name += ANGULAR_VELOCITY if angular_velocity else ""
    return name


if __name__ == '__main__':
    # demo_heuristic_lander(LunarLander(), render=True)
    transforms = [True, True, True, True, True, True]
    demo_heuristic_lander(LunarLanderTransformedEnv(transforms), render=True)
