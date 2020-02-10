class Clock(object):
    def __init__(self, hour, minutes=0):
        self.minutes = '0'*(2-len(str(minutes)))+str(minutes)
        self.hour = '0'*(2-len(str(hour)))+str(hour)
    def __str__(self):
        return self.hour+":"+self.minutes
    def __repr__(self):
        return self.__str__()
    @classmethod
    def at(cls, hour, minutes=0):
        return cls(hour, minutes)
    def __add__(self,minutes):
        time=(int(self.hour)*60+int(self.minutes)+int(minutes))%(24*60)
        return Clock(time//60,time%60)
    def __sub__(self,minutes):
        return self+((-1)*minutes)
    def __eq__(self, other):
        return (self.hour==other.hour and self.minutes==other.minutes)
    def __ne__(self, other):
        return not self==other
