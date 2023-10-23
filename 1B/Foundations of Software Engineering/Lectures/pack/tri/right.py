class TriRight(object):
    def __init__(self, a, b):
        if type(a) == int and type(b) == int:
            self.__init__int(a,b)
        else:
            self.a = a
            self.b = b
            #self.c = (a*a+b*b)**0.5
            mx = max(abs(a), abs(b))
            mn = min(abs(a), abs(b))
            self.c = mx * (1+(mn/mx)**2)**0.5

    def __init__int(self, a, b):
        self.a = a
        self.b = b
        #self.c = (a*a+b*b)**0.5
        mx = max(abs(a), abs(b))
        mn = min(abs(a), abs(b))
        c = mx * (1+(mn/mx)**2)**0.5
        c = int(self.c)
        assert c == self.c