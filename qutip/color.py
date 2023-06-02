from qutip.settings import settings
from matplotlib import cm

__all__ = ['color']


class Color:
    """
    Colormap for visualization functions
    """
    def __init__(self):
        self._cmap = None  # プライベート変数を定義

    @property
    def cmap(self):
        """Set colormap corresponding to colorblind_safe"""
        if settings.colorblind_safe:
            self._cmap = cm.Greys_r
        else:
            self._cmap = cm.RdBu
        return self._cmap

    @cmap.setter
    def cmap(self,value):
        """Set colormap manually"""
        self._cmap = value


color = Color()