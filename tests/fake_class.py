

class FakeClass:
    def __init__(self, param1=None):
        self.param1 = param1

    def test_func(self):
        return True

    def test_func_with_param1(self):
        return self.param1