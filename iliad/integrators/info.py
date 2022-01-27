class Info:
    def __init__(self):
        self.success: bool = True
        self.invalid: bool = False
        self.logdet: bool = 0.0

class EuclideanLeapfrogInfo(Info):
    pass

class RiemannianLeapfrogInfo(Info):
    def __init__(self):
        super().__init__()
        self.num_iters_pos: int = 0
        self.num_iters_mom: int = 0
        self.kl: float = 0.0

class SoftAbsLeapfrogInfo(RiemannianLeapfrogInfo):
    pass

class ImplicitMidpointInfo(Info):
    def __init__(self):
        super().__init__()
        self.num_iters: int = 0

class CoupledInfo(Info):
    def __init__(self):
        super().__init__()
        self.num_iters: int = 0

class LobattoInfo(CoupledInfo):
    pass

class LagrangianLeapfrogInfo(Info):
    pass

class GaussLegendreInfo(CoupledInfo):
    pass
