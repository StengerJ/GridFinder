import heapq
from itertools import count


VALID_MOVE_TUPLES = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}


class AStartResult:
    def __init__(
        self,
        seen: bool,
        goal: tuple[int, int] | None,
        path: list[tuple[int, int]],
        actions: list[int],
        step_cost: int,
        expanded_nodes: int,
    ):
        self.found = seen
        self.goal = goal
        self.path = path
        self.actions = actions
        self.step_cost = step_cost
        self.expanded_nodes = expanded_nodes


def h(cordinate: tuple[int, int], goals: list[tuple[int, int]]) -> int:
    """
    Heuristic function that estimates the cost to reach the nearest goal from the current coordinate.
    It uses the Manhattan distance, which is appropriate for grid-based environments where movement is typically restricted to four cardinal directions. 
    The heuristic is admissible and consistent, ensuring that A* will find the optimal path to the goal.
    """
    return min(abs(cordinate[0] - goal[0]) + abs(cordinate[1] - goal[1]) for goal in goals)


def constructPath(
    start: tuple[int, int],
    goal: tuple[int, int],
    parents: dict[tuple[int, int], tuple[int, int]],
    actions: dict[tuple[int, int], int],
) -> tuple[list[tuple[int, int]], list[int]]:
    """
    Reconstructs the path from the start coordinate to the goal coordinate using the parents and actions dictionaries.
        - parents: A dictionary mapping each coordinate to its parent coordinate in the search tree.
        - actions: A dictionary mapping each coordinate to the action taken to reach it from its parent.
    The function backtracks from the goal to the start, collecting the path and the corresponding actions
    taken to reach each coordinate. Finally, it reverses the path and actions to return them in the correct order from start to goal.
    """
    path = [goal]
    actions = []
    current = goal
    while current != start:
        actions.append(actions[current])
        current = parents[current]
        path.append(current)
    path.reverse()
    actions.reverse()
    return path, actions


def getStart(env) -> tuple[int, int]:
    """
    Extracts the starting position of the agent from the environment. 
    It assumes that the environment has an attribute `agent` with an `initial_position` property that provides the starting coordinates of the agent. The function returns these coordinates as a tuple of integers (x, y).
    """
    pos = env.agent.initial_position
    return int(pos.x), int(pos.y)


def getGoal(env) -> list[tuple[int, int]]:
    """
    returns the goal positions from the environment. 
    It iterates through the state dictionary of the environment, checking for states that are marked as "goal". 
    For each goal state found, it collects its coordinates (column and row) and returns a list of these coordinates as tuples of integers.
    This allows the A* algorithm to know where the goals are located in the environment.
    """
    return [
        (int(col), int(row))
        for (col, row), info in env.state_dict.items()
        if info["type"] == "goal"
    ]


def iterateThroughNeighbors(env, coord: tuple[int, int]):
    """
    Iterates through the neighboring coordinates of the given coordinate in the environment.
    For each valid move (defined in VALID_MOVE_TUPLES), it calculates the next coordinate and checks if it is a valid state (not a hole or wall).
    If the next coordinate is valid, it yields the action and the next coordinate as a tuple. This function is used to explore the neighboring states during the A* search algorithm, allowing it to expand nodes and find paths to the goal.
    """
    for action, (dx, dy) in VALID_MOVE_TUPLES.items():
        next_coord = (coord[0] + dx, coord[1] + dy)
        info = env.state_dict.get(next_coord)
        if info is None or info["type"] == "hole" or info["type"] == "wall":
            continue
        yield action, next_coord


def aStarSearch(env) -> AStartResult:
    start = getStart(env)
    goals = getGoal(env)
    if not goals:
        raise ValueError("The environment does not define a goal state.")

    frontier = []
    order = count()
    heapq.heappush(frontier, (h(start, goals), 0, next(order), start))
    parents: dict[tuple[int, int], tuple[int, int]] = {}
    actions_to: dict[tuple[int, int], int] = {}
    best_g = {start: 0}
    expanded_nodes = 0

    while frontier:
        _, g_cost, _, current = heapq.heappop(frontier)
        if g_cost != best_g.get(current):
            continue

        expanded_nodes += 1
        if current in goals:
            path, actions = constructPath(start, current, parents, actions_to)
            return AStartResult(
                seen=True,
                goal=current,
                path=path,
                actions=actions,
                step_cost=len(actions),
                expanded_nodes=expanded_nodes,
            )

        for action, neighbor in iterateThroughNeighbors(env, current):
            tentative_g = g_cost + 1
            if tentative_g >= best_g.get(neighbor, float("inf")):
                continue
            best_g[neighbor] = tentative_g
            parents[neighbor] = current
            actions_to[neighbor] = action
            priority = tentative_g + h(neighbor, goals)
            heapq.heappush(frontier, (priority, tentative_g, next(order), neighbor))

    return AStartResult(
        seen=False,
        goal=None,
        path=[],
        actions=[],
        step_cost=0,
        expanded_nodes=expanded_nodes,
    )
