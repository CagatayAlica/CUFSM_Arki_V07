class Cases:
    def __init__(self, case: str):
        self.case = case
        self.explanation: str
        if self.case == ' Axial':
            self.explanation = 'Pure axial compression.'
        else:
            self.explanation = 'Bending about its neutral axis and creates compression at top fiber.'
