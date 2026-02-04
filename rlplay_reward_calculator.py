import numpy as np


class RlplayRewardCalculator:
    def __init__(self):
        self.prev_observation = None
        self.observation = None

    def update(self, observation):
        self.prev_observation = self.observation
        self.observation = observation

    def reset(self):
        self.prev_observation = None
        self.observation = None

    def calculate_checkpoint_reward(self, weight=0.0):
        """
        [檢查點獎勵]
        目標：當吃到新的檢查點時，給予一個大的獎勵，鼓勵 AI 往正確路徑前進。

        提示與引導：
        1. 比較 '上一幀的檢查點索引' (self.prev_observation["last_checkpoint_index"])
           與 '這一幀的檢查點索引' (self.observation["last_checkpoint_index"]).
        2. 如果這一幀的索引 > 上一幀的索引，代表進度推進了。
        3. 回傳參數中的 `weight` 作為獎勵。
        4. 如果沒有變化，回傳 0.0。
        """
        if self.prev_observation is None or self.observation is None:
            return 0.0

        prev_index = self.prev_observation.get("last_checkpoint_index", None)
        current_index = self.observation.get("last_checkpoint_index", None)

        if prev_index is None or current_index is None:
            return 0.0

        if current_index > prev_index:
            return weight

        return 0.0

    def calculate_distance_reward(self, close_weight=0.0, leave_weight=0.0 ):
        """
        [距離獎勵]
        目標：引導 AI 時時刻刻都想靠近目標點。這是一個「持續性」的獎勵。

        提示與引導：
        1. 計算 '上一幀與目標的距離' (prev_distance) 和 '這一幀與目標的距離' (current_distance)。
           (提示：使用 numpy.linalg.norm 計算 target_position 的向量長度)
        2. 比較兩者：
           - 如果 current_distance < prev_distance (變近了)：
             代表方向正確，回傳 `close_weight` (通常是正分)。
           - 如果 current_distance > prev_distance (變遠了)：
             代表走錯方向或正在倒車，回傳 `leave_weight` (通常是負分)。
        3. 如果距離沒變，回傳 0.0。
        """
        if self.prev_observation is None or self.observation is None:
            return 0.0

        prev_target = self.prev_observation.get("target_position", None)
        current_target = self.observation.get("target_position", None)

        if prev_target is None or current_target is None:
            return 0.0

        prev_distance = float(np.linalg.norm(prev_target))
        current_distance = float(np.linalg.norm(current_target))

        if current_distance < prev_distance:
            return close_weight
        if current_distance > prev_distance:
            return leave_weight
        return 0.0

    def calculate_health_reward(self, death_weight=0.0, increase_weight=0.0, decrease_weight=0.0  ):
        """
        [血量獎勵]
        目標：教導 AI 生存的重要性，避免撞牆或受傷，並鼓勵吃補包。

        提示與引導：
        1. 首先檢查是否死亡：
           - 如果這一幀的血量 (agent_health) <= 0，回傳 `death_weight` (通常是很大的扣分)。
        2. 如果沒死，比較上一幀與這一幀的血量：
           - 如果血量變少 (受傷)：回傳 `decrease_weight` (扣分)。
           - 如果血量變多 (吃補包)：回傳 `increase_weight` (加分)。
        """
        if self.observation is None:
            return 0.0

        current_health = self.observation.get("agent_health", None)
        if current_health is None:
            return 0.0

        if current_health <= 0:
            return death_weight

        if self.prev_observation is None:
            return 0.0

        prev_health = self.prev_observation.get("agent_health", None)
        if prev_health is None:
            return 0.0

        if current_health > prev_health:
            return increase_weight
        if current_health < prev_health:
            return decrease_weight
        return 0.0

    def calculate_mud_reward(self, threshold=0.0, leave_weight=0.0, close_weight=0.0 ):
        """
        [陷阱/泥巴偵測獎勵]
        目標：教導 AI 使用「雷達」或「網格資訊」來避開危險區域。

        提示與引導：
        1. 取得這一幀的地形網格資訊 (observation["terrain_grid"]).
        2. 檢查車子前方 (例如 grid[3][2] 或 grid[4][2]) 是否有危險地形。
           - Water (terrain_type == -1)
           - Obstacle (terrain_type == 1)
        3. 設定獎勵規則：
           - 如果正前方有危險且距離很近：回傳 `close_weight` (扣分)，告訴它「前面很危險」。
           - (進階) 如果原本前方有泥巴，現在轉向避開了：可以考慮回傳 `leave_weight`。
        4. 簡單版做法：只要偵測到前方有陷阱就扣分。
        """
        if self.observation is None:
            return 0.0

        grid = self.observation.get("terrain_grid", None)
        if grid is None:
            return 0.0

        def get_terrain_type(cell):
            if isinstance(cell, dict):
                return cell.get("terrain_type", None)
            return cell

        def is_hazard(cell):
            terrain_type = get_terrain_type(cell)
            return terrain_type in (-1, 1)

        def cell_distance(cell):
            if isinstance(cell, dict):
                rel = cell.get("relative_position", None)
                if rel is None:
                    return None
                return float(np.linalg.norm(rel))
            return None

        def is_close_hazard(front_grid):
            try:
                cells = [front_grid[3][2], front_grid[4][2]]
                for cell in cells:
                    if not is_hazard(cell):
                        continue
                    if threshold <= 0.0:
                        return True
                    distance = cell_distance(cell)
                    if distance is None or distance <= threshold:
                        return True
                return False
            except (TypeError, IndexError):
                return False

        current_hazard = is_close_hazard(grid)
        if current_hazard:
            return close_weight

        if self.prev_observation is None:
            return 0.0

        prev_grid = self.prev_observation.get("terrain_grid", None)
        if prev_grid is None:
            return 0.0

        if is_close_hazard(prev_grid):
            return leave_weight
        return 0.0

    def calculate_water_reward(self, threshold=0.0, close_weight=0.0):
        """
        [水坑避讓獎勵]
        目標：強化對水坑的避讓行為（水坑=致命）。

        1. 檢查車子前方 (例如 grid[3][2] 或 grid[4][2]) 是否有水坑 (terrain_type == -1)。
        2. 若水坑很近，回傳 close_weight (通常是較大的負分)。
        """
        if self.observation is None:
            return 0.0

        grid = self.observation.get("terrain_grid", None)
        if grid is None:
            return 0.0

        def get_terrain_type(cell):
            if isinstance(cell, dict):
                return cell.get("terrain_type", None)
            return cell

        def cell_distance(cell):
            if isinstance(cell, dict):
                rel = cell.get("relative_position", None)
                if rel is None:
                    return None
                return float(np.linalg.norm(rel))
            return None

        try:
            cells = [grid[3][2], grid[4][2]]
            for cell in cells:
                if get_terrain_type(cell) != -1:
                    continue
                if threshold <= 0.0:
                    return close_weight
                distance = cell_distance(cell)
                if distance is None or distance <= threshold:
                    return close_weight
        except (TypeError, IndexError):
            return 0.0

        return 0.0
