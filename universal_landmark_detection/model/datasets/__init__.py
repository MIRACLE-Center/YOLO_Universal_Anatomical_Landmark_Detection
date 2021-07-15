from .cephalometric import Cephalometric
from .hand import Hand
from .chest import Chest

def get_dataset(s):
    return {
            'cephalometric':Cephalometric,
            'hand':Hand,
            'chest':Chest,
           }[s.lower()]


