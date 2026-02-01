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

    def calculate_checkpoint_reward(self, weight):
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
        # TODO: 請在此處實作檢查點邏輯
        return 0.0

    def calculate_distance_reward(self, close_weight, leave_weight):
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
        # TODO: 請在此處實作距離判斷邏輯
        return 0.0

    def calculate_health_reward(self, death_weight, increase_weight, decrease_weight):
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
        # TODO: 請在此處實作血量獎勵邏輯
        return 0.0

    def calculate_mud_reward(self, threshold, leave_weight, close_weight):
        """
        [陷阱/泥巴偵測獎勵]
        目標：教導 AI 使用「雷達」或「網格資訊」來避開危險區域。

        提示與引導：
        1. 取得這一幀的地形網格資訊 (observation["terrain_grid"]).
        2. 檢查車子前方 (例如 grid[3][2] 或 grid[4][2]) 是否有泥巴 (terrain_type == -1)。
        3. 設定獎勵規則：
           - 如果正前方有泥巴且距離很近：回傳 `close_weight` (扣分)，告訴它「前面很危險」。
           - (進階) 如果原本前方有泥巴，現在轉向避開了：可以考慮回傳 `leave_weight`。
        4. 簡單版做法：只要偵測到前方有陷阱就扣分。
        """
        # TODO: 請在此處實作避障邏輯
        return 0.0
