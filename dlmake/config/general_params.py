from dataclasses import dataclass


@dataclass
class GeneralParams:
    """
    Some classes like optimizers or learning rate schedulers have different parameters for each of the subclasses,
    and combining all possible parameters into setting classes for them would be inconvenient. Instead, this class
    groups together parameters for such classes.

    This class doesn't have any attributes by default, and instead it creates them based on which parameters are written
    in the confguration file, meaning that any setting written there will be an attribute here, with the attribute type
    being the one set in the settings file.
    """

    @classmethod
    def from_dict(cls, **kwargs):
        obj = cls()
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj
