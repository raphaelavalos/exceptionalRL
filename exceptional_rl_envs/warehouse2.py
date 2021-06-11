from typing import Tuple
import math
from gym.utils import seeding
import gym
import numpy as np
from time import sleep

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

MAP_WITH_EXCEPTION = """
WWWWWWWWWWWWW
WEEEEEEEEEEEW
WPEEPEEEPEEPW
WWWWWEEEWWWWW
WP  PEEEP  PW
W           W
W     S     W
WWWWWWWWWWWWW
"""

MIN_JUMP = 1e-7
PENALTY_COLLISION = -.1
REWARD_PICKUP = 10
REWARD_DELIVER = 10
PENALTY_TIME = -0.0
DEFAULT_SPEED_FACTOR = 1.
EXCEPTION_SPEED_FACTOR = 2.

class Agent:

    def __init__(self, warehouse=None, world=None):
        self.x = 0
        self.y = 0
        self.v = 0
        self.max_v = 1.
        self.v_increment = .25
        self.radius = 0.1
        self.orientation = 0
        self._has_package = False
        self.zone = warehouse
        self.world = world
        self.geom = None
        self._collided = False

    def reset(self, x, y):
        self.x, self.y = x, y
        self.v = 0.
        self.orientation = 2
        self._has_package = False

    def update(self, action):
        if action == 0:
            self.v = min(self.max_v, self.v + self.v_increment)
        elif action == 1:
            self.v = max(0., self.v - self.v_increment)
        elif action in [2, 3]:
            delta = 1 if action == 2 else -1
            if self.in_exception:
                delta *= 2
            self.orientation = (self.orientation + delta) % 8

    def get_speed(self):
        return self.zone.get_speed_factor() * self.v

    @property
    def has_collided(self):
        return self._collided

    @has_collided.setter
    def has_collided(self, value):
        assert type(value) is bool
        self._collided = value

    @property
    def has_package(self):
        return self._has_package

    @has_package.setter
    def has_package(self, value):
        assert type(value) is bool
        self._has_package = value

    @property
    def in_exception(self):
        return self.world.warehouse[int(self.x), int(self.y)].is_exception


class Zone:
    _colors = {
        "W": (0., 0., 0.),
        "S": (0., 1., 0.),
        " ": (1., 1., 1.),
        "P": (.5, .5, .5),
        "E": (0., 0., 1.),
    }

    def __init__(self, x1, x2, y1, y2, warehouse):
        self.x1 = min(x1, x2)
        self.x2 = max(x1, x2)
        self.y1 = min(y1, y2)
        self.y2 = max(y1, y2)
        self._accessible = True
        self._start = False
        self._package_zone = False
        self._has_package = False
        self._exception = False
        self._speed_factor = DEFAULT_SPEED_FACTOR
        # self._root = root
        # self._child = []
        # if isinstance(self._root, Zone):
        #     self._root.register(self)
        self.warehouse = warehouse

    @property
    def color(self):
        return Zone._colors[str(self)]

    @property
    def is_start(self):
        return self._start

    @is_start.setter
    def is_start(self, value):
        assert type(value) is bool
        self._start = value

    @property
    def is_wall(self):
        return not self._accessible

    @is_wall.setter
    def is_wall(self, value):
        assert type(value) is bool
        self._accessible = not value

    @property
    def is_package_zone(self):
        return self._package_zone

    @is_package_zone.setter
    def is_package_zone(self, value):
        assert type(value) is bool
        self._package_zone = value

    @property
    def has_package(self):
        return self._has_package

    @has_package.setter
    def has_package(self, value):
        assert type(value) is bool
        if value:
            assert self._package_zone
        self._has_package = value

    @property
    def is_exception(self):
        return self._exception

    @is_exception.setter
    def is_exception(self, value):
        kwargs = {}
        if isinstance(value, tuple):
            value, kwargs = value
        assert type(value) is bool
        self._exception = value
        if not value:
            self._speed_factor = DEFAULT_SPEED_FACTOR
        else:
            self._speed_factor = kwargs.get("speed_factor", DEFAULT_SPEED_FACTOR)


    # def register(self, zone):
    #     self._child.append(zone)

    def __repr__(self):
        if self.is_wall:
            return "W"
        if self.is_start:
            return "S"
        if self.is_package_zone:
            return "P"
        if self.is_exception:
            return "E"
        return " "

    def __contains__(self, item):
        if isinstance(item, Zone):
            return self.x1 <= item.x1 < item.x2 <= self.x2 and self.y1 <= item.y1 < item.x2 <= self.y2
        if isinstance(item, Agent):
            return self.x1 <= item.x <= self.x2 and self.y1 <= item.y <= self.y2

    def get_speed_factor(self):
        return self._speed_factor

    def trajectory(self, agent, dt):
        x, y = agent.x, agent.y
        assert not self.warehouse[int(x), int(y)].is_wall, "agent should not be here {} {}".format(x, y)
        v = agent.v * self._speed_factor
        if v == 0.:
            return 0.
        ort = agent.orientation
        if ort % 2 == 0:
            if ort == 0:
                altx, alty = 1, 0
            elif ort == 2:
                altx, alty = 0, 1
            elif ort == 4:
                altx, alty = -1, 0
            elif ort == 6:
                altx, alty = 0, -1

            next_zone = self.x1 + altx, self.y1 + alty
            west_zone = self.x1 + altx - alty * (1 - altx), self.y1 + alty - altx * (1 - alty)
            east_zone = self.x1 + altx + alty * (1 - altx), self.y1 + alty + altx * (1 - alty)
            d_west = abs(altx) * abs(y - (self.y1 + (1 + altx) // 2)) + \
                     abs(alty) * abs(x - (self.x1 + (1 - alty) // 2))
            d_east = abs(altx) * abs(y - (self.y1 + (1 - altx) // 2)) + \
                     abs(alty) * abs(x - (self.x1 + (1 + alty) // 2))
            west_zone_block = (d_west < agent.radius) and self.warehouse[west_zone].is_wall
            east_zone_block = (d_east < agent.radius) and self.warehouse[east_zone].is_wall
            next_zone_block = self.warehouse[next_zone].is_wall
            dist = abs(altx) * abs(x - (self.x1 + (1 + altx) // 2)) + \
                   abs(alty) * abs(y - (self.y1 + (1 + alty) // 2))
            block = west_zone_block or east_zone_block or next_zone_block
            if block:
                dist -= agent.radius
            dist = max(0., dist)
            if v * dt <= dist:
                agent.x += altx * v * dt
                agent.y += alty * v * dt
                new_dt = 0
            elif block:
                agent.x += altx * dist
                agent.y += alty * dist
                agent.has_collided = True
                new_dt = 0
            else:
                agent.x += altx * (dist + (1 - altx) * MIN_JUMP / 2)
                agent.y += alty * (dist + (1 - alty) * MIN_JUMP / 2)
                new_dt = dt - dist / v
            # return new_dt

        else:
            if ort == 1:
                altx, alty = 1, 1
                dist_w = abs(self.y1 + 1 - y)
                dist_e = abs(self.x1 + 1 - x)
            elif ort == 3:
                altx, alty = -1, 1
                dist_w = abs(self.x1 - x)
                dist_e = abs(self.y1 + 1 - y)
            elif ort == 5:
                altx, alty = -1, -1
                dist_w = abs(self.y1  - y)
                dist_e = abs(self.x1 - x)
            elif ort == 7:
                altx, alty = 1, -1
                dist_w = abs(self.x1 + 1 - x)
                dist_e = abs(self.y1 - y)
            north_zone = self.x1 + altx, self.y1 + alty
            west_zone = self.x1 + (altx - alty) // 2, self.y1 + (altx + alty) // 2
            east_zone = self.x1 + (altx + alty) // 2, self.y1 + (alty - altx) // 2
            north_block = self.warehouse[north_zone].is_wall
            west_block = self.warehouse[west_zone].is_wall
            east_block = self.warehouse[east_zone].is_wall
            if dist_e < dist_w:
                side_block = (north_block or west_block) and (dist_w - dist_e < agent.radius)
                block = east_block or side_block
                side_dist = dist_e
                if east_block:
                    side_dist = dist_e - agent.radius
                elif side_block:
                    side_dist = dist_w - agent.radius

            elif dist_e > dist_w:
                side_block = (north_block or east_block) and (dist_e - dist_w < agent.radius)
                block = side_block or west_block
                side_dist = dist_w
                if west_block:
                    side_dist = dist_w - agent.radius
                elif side_block:
                    side_dist = dist_e - agent.radius
            else:
                # dist_e == dist_w
                block = north_block or east_block or west_block
                side_dist = dist_e
                if block:
                    side_dist -= agent.radius
            side_dist = max(0., side_dist)
            dist = side_dist * math.sqrt(2)
            if v * dt <= dist:
                agent.x += altx * v * dt / math.sqrt(2)
                agent.y += alty * v * dt / math.sqrt(2)
                new_dt = 0
            elif block:
                agent.x += altx * side_dist
                agent.y += alty * side_dist
                agent.has_collided = True
                new_dt = 0
            else:
                agent.x += altx * (side_dist + (1 - altx) * MIN_JUMP / 2)
                agent.y += alty * (side_dist + (1 - alty) * MIN_JUMP / 2)
                new_dt = dt - dist / v

        return new_dt



class World:
    def __init__(self, maps=None, var_pos=0., var_dt=0.):
        self.agent = Agent(world=self)
        self.var_pos = var_pos
        self.var_dt = var_dt
        self.warehouse = np.empty((13, 8), dtype=Zone)
        self.wall_zones = []
        self.start_zones = []
        self.package_zones = []
        self.empty_zones = []
        self.zone_with_package = None
        self.np_random = None
        self.seed()
        self.dt = 1
        self.warehouse_built = False
        self.maps = maps
        self.maps_prob = None
        if self.maps is None:
            self.maps = [MAP]
        elif isinstance(self.maps, str):
            self.maps = [self.maps]
        elif isinstance(self.maps, (list, tuple)) and not isinstance(self.maps[0], str):
            self.maps, self.maps_prob = zip(*self.maps)
        self.current_map = None
        self.build_warehouses()

    def seed(self, seed=None):
        """Set the seed of the random number generator."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def build_warehouse(self, map):
        map = [s for s in map.splitlines() if s]
        warehouse = np.empty((13, 8), dtype=Zone)
        start_zones = []
        wall_zones = []
        package_zones = []
        empty_zones = []
        for i in range(warehouse.shape[0]):
            for j in range(warehouse.shape[1]):
                zone = Zone(i, i+1, j, j+1, warehouse)
                l = map[-j - 1][i]
                if l == "S":
                    zone.is_start = True
                    start_zones.append(zone)
                elif l == "W":
                    zone.is_wall = True
                    wall_zones.append(zone)
                elif l == "P":
                    zone.is_package_zone = True
                    package_zones.append(zone)
                elif l == "E":
                    zone.is_exception = True
                    zone.is_exception = (True, {"speed_factor": EXCEPTION_SPEED_FACTOR})
                else:
                    empty_zones.append(zone)
                warehouse[i, j] = zone
        return warehouse, start_zones, wall_zones, package_zones, empty_zones

    def build_warehouses(self):
        self.warehouses_list, self.start_zones_list, \
        self.wall_zones_list, self.package_zones_list, \
        self.empty_zones_list = zip(*map(self.build_warehouse, self.maps))

    def reset(self):
        # need to select a map
        self.current_map = self.np_random.choice(range(len(self.maps)),
                                                 p=self.maps_prob)
        self.warehouse = self.warehouses_list[self.current_map]
        self.package_zones = self.package_zones_list[self.current_map]
        self.start_zones = self.start_zones_list[self.current_map]
        self.wall_zones = self.wall_zones_list[self.current_map]
        self.empty_zones = self.empty_zones_list[self.current_map]

        self.zone_with_package = self.np_random.choice(self.package_zones)
        for package_zone in self.package_zones:
            package_zone.has_package = package_zone is self.zone_with_package
        start_zone = self.start_zones[0]
        x = (start_zone.x1 + start_zone.x2) / 2
        y = (start_zone.y1 + start_zone.y2) / 2
        self.agent.reset(x, y)

    def act(self, action):
        assert 0 <= action < 5
        self.agent.update(action)
        dt = self.dt
        if self.var_dt > 0.:
            dt = self.np_random.normal(dt, self.var_dt)
        done = False
        reward = 0
        exception = self.agent.in_exception

        while dt > 0:
            i, j = int(self.agent.x), int(self.agent.y)
            dt = self.warehouse[i, j].trajectory(self.agent, dt)
            exception = exception or self.agent.in_exception
        if self.agent.has_collided:
            self.agent.v = 0.
            reward += PENALTY_COLLISION
            self.agent.has_collided = False
        if self.agent in self.zone_with_package and self.zone_with_package.has_package:
            self.agent.has_package = True
            self.zone_with_package.has_package = False
            reward += REWARD_PICKUP
        if self.agent in self.start_zones[0] and self.agent.has_package:
            reward += REWARD_DELIVER
            done = True
        if not done:
            reward += PENALTY_TIME
        return reward, done, exception

    def observe(self):
        dx, dy = 0., 0.
        if self.var_pos > 0:
            dx, dy = self.np_random.normal(0, self.var_pos, 2)
        obs = np.array([
            (self.agent.x + dx) / self.warehouse.shape[0],
            (self.agent.y + dy) / self.warehouse.shape[1],
            self.agent.v / self.agent.max_v,
            self.agent.orientation / 8,
            int(self.agent.has_package),
            (self.zone_with_package.x1 + self.zone_with_package.x2) / (
                        2 * self.warehouse.shape[0]),
            (self.zone_with_package.y1 + self.zone_with_package.y2) / (
                        2 * self.warehouse.shape[1]),
        ], dtype=np.float)
        return obs


class Warehouse(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }


    def __init__(self, **kwargs):
        super().__init__()
        self.world = World(**kwargs)
        self.agent = self.world.agent
        self.warehouse = self.world.warehouse
        self.viewer = None
        self.horizon = 100
        self._time_step = 0
        self.observation_space = gym.spaces.Box(0, 1, shape=(7,))
        self.action_space = gym.spaces.Discrete(5)
        # self.seed()

    def seed(self, seed=None):
        """Set the seed of the random number generator."""
        return self.world.seed(seed)

    def _observe(self):
        return self.world.observe()

    def reset(self):
        self._time_step = 0
        self.world.reset()
        return self._observe()

    def step(self, action):
        # Actions are (Increase speed, Decrease speed, Rotate Left, Rotate Right, Noop)
        reward, done, exception = self.world.act(action)
        self._time_step += 1
        if self._time_step >= self.horizon:
            done = True
        return self._observe(), reward, done, {"exception": exception}

    def render(self, mode='human'):
        box_size = 50
        warehouse = self.world.warehouse
        screen_width = box_size * warehouse.shape[0]
        screen_height = box_size * warehouse.shape[1]
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.filled_polygons = np.empty(warehouse.shape,
                                            dtype=rendering.Transform)
            for i in range(warehouse.shape[0]):
                for j in range(warehouse.shape[1]):
                    box = rendering.FilledPolygon([
                        (i * box_size, j * box_size),
                        (i * box_size, (j + 1) * box_size),
                        ((i + 1) * box_size, (j + 1) * box_size),
                        ((i + 1) * box_size, j * box_size),
                    ])
                    box.set_color(*warehouse[i, j].color)
                    self.viewer.add_geom(box)
                    self.filled_polygons[i, j] = box
            self.agent.geom = rendering.make_circle(
                int(self.agent.radius * box_size))
            self.agent.geom.set_color(1.0, 0., 0.)
            self.agent.transform = rendering.Transform()
            self.agent.geom.add_attr(self.agent.transform)
            self.viewer.add_geom(self.agent.geom)
        self.agent.transform.set_translation(self.agent.x * box_size,
                                             self.agent.y * box_size)

        for package_zone in self.world.package_zones:
            color = package_zone.color
            if package_zone.has_package:
                color = (0., 0., 1.)
            self.filled_polygons[
                package_zone.x1, package_zone.y1].set_color(*color)

        return self.viewer.render(mode == "rgb_array")

