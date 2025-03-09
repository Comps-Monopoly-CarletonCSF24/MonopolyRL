Actions = ['buy', 'sell', 'do_nothing']
Total_Actions = len(Actions)
class Action:
    def __init__(self, action_type: str):
        self.action_type = action_type
        self.action_index = Actions.index(action_type)
    def __str__(self):
        return self.action_type