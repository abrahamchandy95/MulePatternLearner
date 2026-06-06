from typing import override

from pyTigerGraph import TigerGraphConnection

from mule_pattern_learner.tigergraph.settings import Settings


class Client:
    """
    The underlying `pyTigerGraph.TigerGraphConnection` is exposed
    via the `conn` attribute for downstream modules that need to invoke
    pyTigerGraph methods directly.
    """

    _settings: Settings
    conn: TigerGraphConnection

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.conn = TigerGraphConnection(
            host=settings.host,
            graphname=settings.graphname,
            gsqlSecret=settings.secret.get_secret_value(),
        )
        _ = self.conn.getToken(settings.secret.get_secret_value())

    @property
    def graphname(self) -> str:
        """Name of the graph this client is connected to."""
        return self._settings.graphname

    @override
    def __repr__(self) -> str:
        return f"Client(graphname={self._settings.graphname!r})"
