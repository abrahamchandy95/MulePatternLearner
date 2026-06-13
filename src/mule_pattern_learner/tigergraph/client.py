from functools import partial
import socket
from typing import cast, override

import requests

from pyTigerGraph import TigerGraphConnection

from mule_pattern_learner.tigergraph.settings import Settings

_READ_TIMEOUT_S = 600.0
_CONNECT_TIMEOUT_S = 30.0


class Client:
    """
    Client that connects to TigerGraph
    """

    _settings: Settings
    conn: TigerGraphConnection

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        socket.setdefaulttimeout(_READ_TIMEOUT_S)
        self.conn = TigerGraphConnection(
            host=settings.host,
            graphname=settings.graphname,
            gsqlSecret=settings.secret.get_secret_value(),
        )
        _ = self.conn.getToken(settings.secret.get_secret_value())
        self._install_default_timeout()

    def _install_default_timeout(self) -> None:
        session = cast(object, getattr(self.conn, "_session", None))
        if not isinstance(session, requests.Session):
            return
        if getattr(session.request, "_has_default_timeout", False):
            return
        wrapped = partial(session.request, timeout=(_CONNECT_TIMEOUT_S, _READ_TIMEOUT_S))
        setattr(wrapped, "_has_default_timeout", True)
        setattr(session, "request", wrapped)

    @property
    def graphname(self) -> str:
        return self._settings.graphname

    @override
    def __repr__(self) -> str:
        return f"Client(graphname={self._settings.graphname!r})"
