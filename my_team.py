# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
from collections import deque

import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    agents = {
        'OffensiveReflexAgent': OffensiveReflexAgent,
        'DefensiveReflexAgent': DefensiveReflexAgent,
    }
    return [agents[first](first_index), agents[second](second_index)]



# Infraestructura del atacante de V1

class ReflexCaptureAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        super().register_initial_state(game_state)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor


# Infraestructura independiente para el defensor de V2

class BaseAgent(CaptureAgent):
    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        super().register_initial_state(game_state)
        walls = game_state.get_walls()
        mid = walls.width // 2
        home_x, away_x = (mid - 1, mid) if self.red else (mid, mid - 1)
        self.crossings = [
            ((home_x, y), (away_x, y))
            for y in range(1, walls.height - 1)
            if not walls[home_x][y] and not walls[away_x][y]
        ]
        self.home_boundary = [home for home, _ in self.crossings]

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor

    def distance(self, a, b):
        return self.get_maze_distance(a, b)


# Ataque V1: mejor en las pruebas de ablacion

class OffensiveReflexAgent(ReflexCaptureAgent):
    """Collect food; reroute around a guarded crossing and avoid oscillation."""

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        walls = game_state.get_walls()
        mid = walls.width // 2
        home_x = mid - 1 if self.red else mid
        away_x = mid if self.red else mid - 1
        self.crossings = [
            ((home_x, y), (away_x, y))
            for y in range(1, walls.height - 1)
            if not walls[home_x][y] and not walls[away_x][y]
        ]
        self.home_boundary = [p for p, _ in self.crossings]
        self.cross_target = None
        self.recent_positions = deque(maxlen=12)
        self.last_carrying = 0

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return Directions.STOP

        current = game_state.get_agent_state(self.index)
        position = current.get_position()
        carrying = current.num_carrying
        self.recent_positions.append(position)

        foods = self.get_food(game_state).as_list()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        dangerous = [e.get_position() for e in enemies
                     if e.get_position() is not None
                     and not e.is_pacman and e.scared_timer <= 1]

        # A predictable defender near the middle can freeze a reflex attacker.
        # Choose a safe crossing BEFORE entering enemy territory, rather than
        # always heading to the closest food through the same passage.
        if not current.is_pacman and carrying == 0 and foods and self.crossings:
            self.cross_target = self._select_crossing(position, foods, dangerous)
        elif current.is_pacman:
            self.cross_target = None

        # Bring food home before the end; carry limit can be reduced if threatened.
        threatened = current.is_pacman and any(
            self.get_maze_distance(position, g) <= 4 for g in dangerous
        )
        returning = carrying >= 5 or (carrying >= 2 and threatened)
        if len(foods) <= 2 and carrying > 0:
            returning = True

        if returning and self.home_boundary:
            target = min(self.home_boundary,
                         key=lambda p: self.get_maze_distance(position, p))
        elif not current.is_pacman and self.cross_target is not None:
            target = self.cross_target[1]  # cross onto enemy side
        elif foods:
            # Prefer food reachable without getting too close to a ghost.
            target = min(foods, key=lambda p: self.get_maze_distance(position, p)
                         + 8 * self._ghost_risk(p, dangerous))
        else:
            target = self.start

        # Repeated visits signal a local cycle. Temporarily favor moves that
        # explore less-visited squares, while keeping ghost safety paramount.
        counts = {p: self.recent_positions.count(p) for p in self.recent_positions}
        rated = []
        for action in actions:
            successor = self.get_successor(game_state, action)
            new_state = successor.get_agent_state(self.index)
            new_pos = new_state.get_position()
            score = -self.get_maze_distance(new_pos, target)
            score -= 2.5 * counts.get(new_pos, 0)
            if action == Directions.STOP:
                score -= 12

            # Strong but finite ghost avoidance. Use opponent side proximity
            # even before crossing, so border guards influence route choice.
            risk = self._ghost_risk(new_pos, dangerous)
            if new_state.is_pacman:
                score -= 35 * risk
            else:
                score -= 3 * risk

            # Banked points are preferable when already carrying food.
            if current.is_pacman and not new_state.is_pacman and carrying > 0:
                score += 30 * carrying

            # Prefer picking up a pellet on an otherwise equally good move.
            if len(self.get_food(successor).as_list()) < len(foods):
                score += 15
            rated.append((score, action))

        best = max(value for value, _ in rated)
        return random.choice([action for value, action in rated if value == best])

    def _ghost_risk(self, pos, ghosts):
        if not ghosts:
            return 0
        d = min(self.get_maze_distance(pos, ghost) for ghost in ghosts)
        return max(0, 5 - d)

    def _select_crossing(self, position, foods, ghosts):
        # Retain existing safe crossing to avoid flip-flopping between gates.
        if self.cross_target is not None:
            home, away = self.cross_target
            if all(self.get_maze_distance(away, g) >= 5 for g in ghosts):
                return self.cross_target

        def cost(crossing):
            home, away = crossing
            approach = self.get_maze_distance(position, home)
            remaining = min(self.get_maze_distance(away, f) for f in foods)
            risk = self._ghost_risk(away, ghosts)
            # Spread the attack toward openings not watched by visible ghosts.
            return approach + 0.5 * remaining + 20 * risk

        return min(self.crossings, key=cost)


# Defensa V2: mejor en las pruebas de ablacion

class DefensiveReflexAgent(BaseAgent):
    """Border defense, with memory of disappeared defended food."""

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.previous_food = set(self.get_food_you_are_defending(game_state).as_list())
        self.previous_capsules = set(self.get_capsules_you_are_defending(game_state))
        self.investigate = None
        self.investigate_turns = 0
        self.patrol = None
        if self.home_boundary:
            mid_y = game_state.get_walls().height // 2
            self.patrol = min(self.home_boundary, key=lambda p: abs(p[1] - mid_y))

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return Directions.STOP
        my_state = game_state.get_agent_state(self.index)
        pos = my_state.get_position()

        current_food = set(self.get_food_you_are_defending(game_state).as_list())
        current_capsules = set(self.get_capsules_you_are_defending(game_state))
        missing = (self.previous_food - current_food) | (self.previous_capsules - current_capsules)
        self.previous_food = current_food
        self.previous_capsules = current_capsules
        if missing:
            self.investigate = min(missing, key=lambda p: self.distance(pos, p))
            self.investigate_turns = 14
        elif self.investigate_turns > 0:
            self.investigate_turns -= 1
        if self.investigate == pos:
            self.investigate_turns = 0

        opponents = [game_state.get_agent_state(i)
                     for i in self.get_opponents(game_state)]
        invaders = [e.get_position() for e in opponents
                    if e.is_pacman and e.get_position() is not None]

        if invaders:
            target = min(invaders, key=lambda p: self.distance(pos, p))
            weight = 10
        elif self.investigate is not None and self.investigate_turns > 0:
            target = self.investigate
            weight = 11
        else:
            target = self.patrol or self.start
            weight = 5

        original_direction = my_state.configuration.direction
        reverse = Directions.REVERSE[original_direction]
        rated = []
        for action in actions:
            successor = self.get_successor(game_state, action)
            new_state = successor.get_agent_state(self.index)
            next_pos = new_state.get_position()
            value = -weight * self.distance(next_pos, target)
            if new_state.is_pacman:
                value -= 1000
            if action == Directions.STOP:
                value -= 100
            if action == reverse:
                value -= 2
            if invaders and my_state.scared_timer > 0:
                # Scared ghosts are vulnerable; keep away from an invader.
                closest = min(self.distance(next_pos, p) for p in invaders)
                if closest <= 1:
                    value -= 200
            rated.append((value, action))

        best = max(v for v, _ in rated)
        return random.choice([a for v, a in rated if v == best])

