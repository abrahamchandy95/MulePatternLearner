"""
TigerGraph (Savanna) connection.

Two responsibilities, deliberately separated (SRP):

  * ``TigerGraphConfig`` — the connection PARAMETERS (host, graph, secret) and
    nothing else. It changes for exactly one reason: the credentials / target
    instance change. It is plain, frozen, typed data. It carries no token
    knobs, no query helpers, no unrelated settings.

  * ``connect()`` — the FACTORY that turns a config into a live, authenticated
    ``TigerGraphConnection``. It changes for a different reason: the Savanna
    authentication flow (secret -> bearer token) changes. Consumers that *use*
    the connection (the masking runner, the PyG loader, writeback) live in
    other modules and depend only on the returned client.

Usage
-----
    from mule_pattern_learner.tigergraph.connection import (
        TigerGraphConfig, connect,
    )

    conn = connect(TigerGraphConfig.from_env())   # reads .env
    # conn is a ready pyTigerGraph TigerGraphConnection with a token set.

Credentials come from environment variables (loaded from a gitignored ``.env``
via python-dotenv), matching the names used in this project's ``.env``:

    HOST=https://tg-<cluster-uuid>.tg-<n>.i.tgcloud.io
    GRAPHNAME=Mule_Pattern_Learner
    SECRET=<secret created with `CREATE SECRET` while that graph is active>

A Savanna secret is bound to one (user, graph) pair, so each graph/project gets
its own secret in its own ``.env`` — never reuse one across graphs or
workspaces.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv
from pyTigerGraph import TigerGraphConnection


# Environment variable names this project uses (match the .env file).
_ENV_HOST = "HOST"
_ENV_GRAPHNAME = "GRAPHNAME"
_ENV_SECRET = "SECRET"


@dataclass(frozen=True, slots=True)
class TigerGraphConfig:
    """Connection parameters for a single TigerGraph graph.

    Holds only what identifies and authenticates the connection. One reason to
    change: the credentials or target instance.

    Attributes
    ----------
    host:
        Full instance host URL, e.g.
        ``https://tg-<cluster-uuid>.tg-<n>.i.tgcloud.io``. This is the per-
        workspace REST host (NOT the ``tools.tgcloud.io`` GraphStudio URL).
    graphname:
        The graph to operate on (e.g. ``Mule_Pattern_Learner``).
    secret:
        Savanna secret, exchanged for a bearer token at connect time. Bound to
        this (user, graph); do not share across graphs/workspaces.
    """

    host: str
    graphname: str
    secret: str

    @classmethod
    def from_env(cls, *, dotenv_path: str | None = None) -> TigerGraphConfig:
        """Build a config from environment variables.

        Loads a ``.env`` file if present (python-dotenv), then reads HOST,
        GRAPHNAME, and SECRET. Loading-from-env is just a way of constructing
        the config, so it lives here rather than in the factory.

        Parameters
        ----------
        dotenv_path:
            Optional explicit path to a ``.env`` file. If None, python-dotenv's
            default discovery is used (searches up from the CWD). Existing
            environment variables are not overridden.

        Raises
        ------
        ValueError:
            If any of HOST / GRAPHNAME / SECRET is missing or empty, with a
            message naming the offending variable(s).
        """
        load_dotenv(dotenv_path=dotenv_path, override=False)

        host = os.environ.get(_ENV_HOST, "").strip()
        graphname = os.environ.get(_ENV_GRAPHNAME, "").strip()
        secret = os.environ.get(_ENV_SECRET, "").strip()

        missing = [
            name
            for name, value in (
                (_ENV_HOST, host),
                (_ENV_GRAPHNAME, graphname),
                (_ENV_SECRET, secret),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ". Set them in your .env (see .env.example)."
            )

        return cls(host=host, graphname=graphname, secret=secret)


def connect(
    config: TigerGraphConfig, *, token_lifetime_seconds: int | None = None
) -> TigerGraphConnection:
    """Create an authenticated TigerGraph connection from a config.

    The factory's single responsibility is the Savanna auth flow: construct the
    client for ``config.host`` / ``config.graphname``, then exchange
    ``config.secret`` for a REST++ bearer token (``getToken``) and attach it.

    Token lifetime is intentionally NOT part of ``TigerGraphConfig`` — it is an
    auth-flow detail, not a connection identity parameter. It defaults to
    pyTigerGraph's default (~30 days) and can be overridden per call here.

    Parameters
    ----------
    config:
        The connection parameters.
    token_lifetime_seconds:
        Optional token lifetime in seconds. If None, pyTigerGraph's default is
        used.

    Returns
    -------
    TigerGraphConnection
        A connection with a valid token set, ready for queries / loaders.
    """
    conn = TigerGraphConnection(host=config.host, graphname=config.graphname)

    if token_lifetime_seconds is None:
        conn.getToken(config.secret)
    else:
        conn.getToken(config.secret, lifetime=token_lifetime_seconds)

    return conn
