from typing import Tuple
import math
from gym.utils import seeding
import gym
import numpy as np

MAP = """
WWWWWWWWWWWWW
W           W
WP  P   P  PW
WWWWW   WWWWW
WP  P   P  PW
W           W
W     S     W
WWWWWWWWWWWWW
"""


class Zone:
    def __init__(self, x1, x2, y1, y2, root=None):
        self.x1 = min(x1, x2)
        self.x2 = max(x1, x2)
        self.y1 = min(y1, y2)
        self.y2 = max(y1, y2)
        self.accessible = True
        self.start = False
        self._speed_factor = 1.
        self._root = root
        self._child = []
        if isinstance(self._root, Zone):
            self._root.register(self)
        self.neighbours = {
            "N": None,
            "NE": None,
            "E": None,
            "SE": None,
            "S": None,
            "SW": None,
            "W": None,
            "NW": None,
        }

    def register(self, zone):
        self._child.append(zone)

    def __repr__(self):
        return self.__class__.__name__

    def __contains__(self, item):
        if isinstance(item, Zone):
            return self.x1 <= item.x1 < item.x2 <= self.x2 and self.y1 <= item.y1 < item.x2 <= self.y2
        if isinstance(item, Agent):
            return self.x1 <= item.x <= self.x2 and self.y1 <= item.y <= self.y2

    def get_speed_factor(self):
        return self._speed_factor


class ExceptionZone(Zone):

    def __init__(self, x1, x2, y1, y2):
        super().__init__(x1, x2, y1, y2)
        self._active = False

    def is_active(self):
        return self._active


class PackageZone(Zone):
    def __init__(self, x1, x2, y1, y2, root):
        super().__init__(x1, x2, y1, y2, root)
        self._has_package = False

    @property
    def has_package(self):
        return self._has_package

    @has_package.setter
    def has_package(self, value):
        self._has_package = value


class StartZone(Zone):
    def __init__(self, x1, x2, y1, y2, root):
        super().__init__(x1, x2, y1, y2, root)
        self.start = True


class Wall(Zone):

    def __init__(self, x1, x2, y1, y2, root):
        super().__init__(x1, x2, y1, y2, root)
        self.accessible = False


letter_to_zone = {
    "W": Wall,
    " ": Zone,
    "P": PackageZone,
    "S": StartZone,
}

zone_to_color = {
    Wall: (0., 0., 0.),
    StartZone: (0., 1., 0),
    Zone: (1., 1., 1.),
    PackageZone: (.5, .5, .5),
}



class Warehouse(gym.Env):
    def __init__(self):
        super().__init__()
        self.warehouse = np.empty((13, 8), dtype=Zone)
        self.start_zones = []
        self.walls_zones = []
        self.package_zones = []
        self.zone_with_package = None
        self.empty_zones = []
        self.build_warehouse()
        self.agent = Agent(self)
        self.viewer = None
        self.np_random = None
        self.observation_space = gym.spaces.Box(0, 1, shape=(7,))
        self.action_space = gym.spaces.Discrete(5)
        self.seed()

    def seed(self, seed=None):
        """Set the seed of the random number generator."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _observe(self):

        obs = np.array([
            self.agent.x / self.warehouse.shape[0],
            self.agent.y / self.warehouse.shape[1],
            self.agent.v / self.agent.max_v,
            self.agent.orientation / 8,
            int(self.agent.carry_package),
            (self.zone_with_package.x1 + self.zone_with_package.x2) / (2 * self.warehouse.shape[0]),
            (self.zone_with_package.y1 + self.zone_with_package.y2) / (2 * self.warehouse.shape[1]),
        ], dtype=np.float)
        return obs

    def reset(self):
        # Choose start zone
        # start_zone = np.random.choice(self.start_zones)
        # x = start_zone.x1 + np.random.random() * (start_zone.x2 - start_zone.x1)
        # y = start_zone.y1 + np.random.random() * (start_zone.y2 - start_zone.y1)
        start_zone = self.start_zones[0]
        x = (start_zone.x1 + start_zone.x2) / 2
        y = (start_zone.y1 + start_zone.y2) / 2
        self.agent.reset(x, y)
        self.zone_with_package = self.np_random.choice(self.package_zones)
        for package_zone in self.package_zones:
            package_zone.has_package = package_zone is self.zone_with_package
        return self._observe()

    def step(self, action):
        # Actions are (Increase speed, Decrease speed, Rotate Left, Rotate Right, Noop)
        reward = -0.0
        done = False
        self.agent.update(action)
        dt = 1
        while dt > 0:
            x, y, orientation = self.agent.x, self.agent.y, self.agent.orientation
            i, j = int(x), int(y)
            zone = self.warehouse[i, j]
            assert zone.accessible, "Agent at {},{} but shouldn't be there !".format(x, y)
            if self.agent.v == 0.0:
                break
            if orientation == 0:
                d = i + 1 - x
                next_tile = (i + 1, j)
            elif orientation == 1:
                d = math.sqrt(2 * min(i + 1 - x, j + 1 - y) ** 2)
                next_tile = (i + 1, j) if i + 1 - x < j + 1 - y else (i, j + 1)
            elif orientation == 2:
                d = j + 1 - y
                next_tile = (i, j + 1)
            elif orientation == 3:
                d = math.sqrt(2 * min(x - i, j + 1 - y) ** 2)
                next_tile = (i - 1, j) if x - i < j + 1 - y else (i, j + 1)
            elif orientation == 4:
                d = x - i
                next_tile = (i - 1, j)
            elif orientation == 5:
                d = math.sqrt(2 * min(x - i, y - j) ** 2)
                next_tile = (i - 1, j) if x - i < y - j else (i, j - 1)
            elif orientation == 6:
                d = y - j
                next_tile = (i, j - 1)
            elif orientation == 7:
                d = math.sqrt(2 * min(i + 1 - x, y - j) ** 2)
                next_tile = (i + 1, j) if i + 1 - x < y - j else (i, j - 1)
            if d > 0.:
                d_ = d - (1 - self.warehouse[next_tile].accessible) * self.agent.radius
                # Will it make it to the next tile?
                agent_max_distance = self.agent.v * zone.get_speed_factor() * dt
                # compute dt that we can apply
                if agent_max_distance <= d_:
                    dt_ = dt
                else:
                    dt_ = d_ / (self.agent.v * zone.get_speed_factor())
            else:
                print("agent at {},{}".format(x,y))
                dt_ = min(1e-4, dt)
            # dt_ = agent_max_distance / min(agent_max_distance, d_)
            self.agent.x += math.cos(2 * math.pi * orientation / 8) * self.agent.v * zone.get_speed_factor() * dt_
            self.agent.y += math.sin(2 * math.pi * orientation / 8) * self.agent.v * zone.get_speed_factor() * dt_
            if dt_ < dt:
                if self.warehouse[next_tile].accessible:
                    dt = dt - dt_
                else:
                    self.agent.v = 0
                    dt = 0
            else:
                dt = 0
        if isinstance(zone, PackageZone) and zone.has_package:
            self.agent.carry_package = True
            zone.has_package = False
            reward = 1
        if isinstance(zone, StartZone) and self.agent.carry_package:
            done = True
            reward = 1
        return self._observe(), reward, done, {}

    def build_warehouse(self):
        map = [s for s in MAP.splitlines() if s]
        for i in range(self.warehouse.shape[0]):
            for j in range(self.warehouse.shape[1]):
                zone = letter_to_zone[map[-j - 1][i]](i, i + 1, j, j + 1, self)
                self.warehouse[i, j] = zone
                if isinstance(zone, Wall):
                    self.walls_zones.append(zone)
                elif isinstance(zone, PackageZone):
                    self.package_zones.append(zone)
                elif isinstance(zone, StartZone):
                    self.start_zones.append(zone)
                else:
                    self.empty_zones.append(zone)

    def render(self, mode='human'):
        box_size = 50
        screen_width = box_size * self.warehouse.shape[0]
        screen_height = box_size * self.warehouse.shape[1]
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.filled_polygons = np.empty(self.warehouse.shape, dtype=rendering.Transform)
            for i in range(self.warehouse.shape[0]):
                for j in range(self.warehouse.shape[1]):
                    box = rendering.FilledPolygon([
                        (i * box_size, j * box_size),
                        (i * box_size, (j + 1) * box_size),
                        ((i + 1) * box_size, (j + 1) * box_size),
                        ((i + 1) * box_size, j * box_size),
                    ])
                    box.set_color(*zone_to_color[type(self.warehouse[i, j])])
                    self.viewer.add_geom(box)
                    self.filled_polygons[i, j] = box
            self.agent.geom = rendering.make_circle(int(self.agent.radius * box_size))
            self.agent.geom.set_color(1.0, 0., 0.)
            self.agent.transform = rendering.Transform()
            self.agent.geom.add_attr(self.agent.transform)
            self.viewer.add_geom(self.agent.geom)
        self.agent.transform.set_translation(self.agent.x * box_size, self.agent.y * box_size)

        for package_zone in self.package_zones:
            color = zone_to_color[PackageZone]
            if package_zone.has_package:
                color = (0., 0., 1.)
            self.filled_polygons[package_zone.x1, package_zone.y1].set_color(*color)

        return self.viewer.render(False)


class Agent:

    def __init__(self, warehouse):
        self.x = 0
        self.y = 0
        self.v = 0
        self.max_v = 1.
        self.v_increment = 0.1
        self.radius = 0.1
        self.orientation = 0
        self.carry_package = False
        self.zone = warehouse
        self.geom = None

    def reset(self, x, y):
        self.x, self.y = x, y
        self.v = 0.
        self.orientation = 2
        self.carry_package = False

    def update(self, action):
        if action == 0:
            self.v = min(self.max_v, self.v + self.v_increment)
        elif action == 1:
            self.v = max(0., self.v - self.v_increment)
        elif action == 2:
            self.orientation = (self.orientation + 1) % 8
        elif action == 3:
            self.orientation = (self.orientation - 1) % 8

    def get_speed(self):
        return self.zone.get_speed_factor() * self.v
