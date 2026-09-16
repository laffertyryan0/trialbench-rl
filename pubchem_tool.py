"""PubChem lookup tool for RL rollouts, modeled directly on
tinker_cookbook.recipes.search_tool.tools.ChromaTool — same shape
(stateless-ish, shared across trajectories, @tool-decorated method).

Uses PubChem's PUG REST API (public, no key/license needed):
https://pubchem.ncbi.nlm.nih.gov/rest/pug/
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from typing import Annotated

from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool

logger = logging.getLogger(__name__)

_PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_CONNECTION_SEMAPHORE = asyncio.Semaphore(16)  # be polite to PubChem's rate limits

_PROPERTIES = "MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount"


class PubChemTool:
    """Look up chemical/drug properties by SMILES or name via PubChem."""

    def __init__(self, max_retries: int = 3, timeout: float = 15.0):
        self._max_retries = max_retries
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def _get(self, url: str) -> httpx.Response | None:
        client = await self._ensure_client()
        for attempt in range(self._max_retries):
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    return r
                if r.status_code == 404:
                    return r  # not found, don't retry
                await asyncio.sleep(1.5 * (attempt + 1))
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                logger.warning(f"PubChem request failed (attempt {attempt + 1}): {e}")
                await asyncio.sleep(1.5 * (attempt + 1))
        return None

    @tool
    async def lookup_drug(
        self,
        identifier: Annotated[
            str, "A drug's SMILES string or common/generic name to look up."
        ],
        by: Annotated[str, "Either 'smiles' or 'name' — how to interpret `identifier`."] = "smiles",
    ) -> ToolResult:
        """Look up a compound's chemical properties (molecular weight, formula,
        IUPAC name, LogP, polar surface area, H-bond donor/acceptor counts) on
        PubChem, given its SMILES structure or common name."""
        async with _CONNECTION_SEMAPHORE:
            domain = "smiles" if by == "smiles" else "name"
            # PubChem needs the identifier URL-escaped in the path.
            import urllib.parse

            escaped = urllib.parse.quote(identifier, safe="")
            url = f"{_PUBCHEM_BASE}/compound/{domain}/{escaped}/property/{_PROPERTIES}/JSON"
            resp = await self._get(url)

        if resp is None:
            return simple_tool_result(f"PubChem lookup failed for '{identifier}' (network error).")
        if resp.status_code == 404:
            return simple_tool_result(
                f"No PubChem compound found for '{identifier}' (by {by}). "
                "It may be a mixture, an invalid SMILES, or not indexed under that name."
            )
        try:
            data = resp.json()
            props = data["PropertyTable"]["Properties"][0]
        except Exception:
            return simple_tool_result(f"PubChem returned an unparseable response for '{identifier}'.")

        lines = [f"PubChem result for '{identifier}':"]
        for k, v in props.items():
            if k == "CID":
                continue
            lines.append(f"  {k}: {v}")
        return simple_tool_result("\n".join(lines))
