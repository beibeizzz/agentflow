from __future__ import annotations

import asyncio
import hashlib
import html
import http.client
import ipaddress
import re
import socket
import ssl
import tempfile
from time import monotonic
import urllib.error
import urllib.parse
from html.parser import HTMLParser

from agentflow_rl.runtime.errors import InfrastructureError

from .contracts import PageDocument
from .concurrency import CrossProcessSlotLimiter


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._in_title = False
        self._ignored_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "li", "h1", "h2", "h3", "br"} and not self._ignored_depth:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        value = data.strip()
        if not value:
            return
        if self._in_title:
            self.title = f"{self.title} {value}".strip()
        else:
            self.parts.append(value)


def _resolve_public_url(url: str):
    if any(ord(char) < 32 or ord(char) == 127 for char in url):
        raise ValueError("page URL contains control characters")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("page URL must use HTTP or HTTPS with a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("page URL must omit credentials")
    try:
        default_port = 443 if parsed.scheme == "https" else 80
        addresses = socket.getaddrinfo(
            parsed.hostname,
            parsed.port or default_port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise InfrastructureError("page hostname resolution failed") from exc
    if not addresses:
        raise InfrastructureError("page hostname resolution returned no addresses")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if (
            not ip.is_global
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_unspecified
            or ip.is_multicast
        ):
            raise ValueError("page URL resolves to a restricted network address")
    return parsed, addresses


def validate_public_url(url: str) -> str:
    parsed, _ = _resolve_public_url(url)
    return urllib.parse.urlunsplit(parsed)


def _connect_validated(addresses, timeout_s: float):
    """Connect using numeric sockaddr values from this request's DNS check."""
    last_error = None
    for family, socktype, proto, _, sockaddr in addresses:
        sock = socket.socket(family, socktype, proto)
        try:
            sock.settimeout(timeout_s)
            sock.connect(sockaddr)
            if ipaddress.ip_address(sock.getpeername()[0]) != ipaddress.ip_address(sockaddr[0]):
                raise OSError("page connection peer differs from validated address")
            return sock
        except OSError as exc:
            sock.close()
            last_error = exc
    raise OSError("page connection failed") from last_error


def _fetch(
    url: str, *, timeout_s: float, max_bytes: int, max_redirects: int
) -> tuple[str, str, int]:
    current = url
    for _ in range(max_redirects + 1):
        parsed, addresses = _resolve_public_url(current)
        # HTTPConnection retains the hostname for Host. Supplying the socket
        # prevents its normal second DNS lookup and bypasses environment proxies.
        connection = http.client.HTTPConnection(
            parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80),
            timeout=timeout_s,
        )
        try:
            connection.sock = _connect_validated(addresses, timeout_s)
            if parsed.scheme == "https":
                connection.sock = ssl.create_default_context().wrap_socket(
                    connection.sock, server_hostname=parsed.hostname,
                )
            target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
            connection.request("GET", target, headers={
                "User-Agent": "AgentFlowBeta/0.1",
                "Host": parsed.netloc.encode("idna").decode("ascii"),
            })
            with connection.getresponse() as response:
                if response.status in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    if not location:
                        raise ValueError("redirect response has no Location")
                    current = urllib.parse.urljoin(current, location)
                    continue
                if response.status >= 400:
                    raise urllib.error.HTTPError(current, response.status, response.reason, response.headers, None)
                content_type = response.headers.get_content_type()
                if content_type not in {"text/html", "text/plain"}:
                    raise ValueError(f"unsupported page content type: {content_type}")
                payload = response.read(max_bytes + 1)
                if len(payload) > max_bytes:
                    raise ValueError("page exceeds the configured size limit")
                charset = response.headers.get_content_charset() or "utf-8"
                return current, payload.decode(charset, errors="replace"), len(payload)
        finally:
            connection.close()
    raise ValueError("page exceeds the configured redirect limit")


def extract_page(raw_html: str) -> tuple[str, tuple[str, ...]]:
    parser = _TextExtractor()
    parser.feed(raw_html)
    title = html.unescape(parser.title).strip() or "Untitled page"
    text = html.unescape(" ".join(parser.parts))
    text = re.sub(r"[ \t]+", " ", text)
    paragraphs = tuple(
        line.strip() for line in re.split(r"\n+", text) if line.strip()
    )
    return title, paragraphs


class WebPageReaderBackend:
    revision = "safe-web-reader-v2"

    def __init__(
        self,
        *,
        timeout_s: float = 20.0,
        max_bytes: int = 2_000_000,
        max_redirects: int = 3,
        max_concurrency: int = 12,
        coordination_dir: str | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.requests = 0
        self.queue_wait_ms = 0.0
        self.fetch_ms = 0.0
        self.extraction_ms = 0.0
        self.fetched_bytes = 0
        self.limiter = CrossProcessSlotLimiter(
            coordination_dir or f"{tempfile.gettempdir()}/agentflow-locks",
            name="web-page-reader",
            limit=max_concurrency,
        )

    async def open(self, *, url: str, result_id: str | None = None) -> PageDocument:
        try:
            async with self.limiter.slot(timeout_s=self.timeout_s) as wait_ms:
                self.requests += 1
                self.queue_wait_ms += wait_ms
                fetch_started = monotonic()
                final_url, raw, payload_bytes = await asyncio.to_thread(
                    _fetch,
                    url,
                    timeout_s=self.timeout_s,
                    max_bytes=self.max_bytes,
                    max_redirects=self.max_redirects,
                )
                self.fetch_ms += (monotonic() - fetch_started) * 1000.0
                self.fetched_bytes += payload_bytes
                extraction_started = monotonic()
                title, passages = await asyncio.to_thread(extract_page, raw)
                self.extraction_ms += (monotonic() - extraction_started) * 1000.0
        except ValueError:
            raise
        except Exception as exc:
            raise InfrastructureError("page read failed") from exc
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return PageDocument(
            result_id=result_id or digest[:16],
            canonical_url=final_url,
            title=title,
            passages=passages,
            content_sha256=digest,
        )

    def metrics(self) -> dict[str, float]:
        return {
            "requests": float(self.requests),
            "queue_wait_ms": self.queue_wait_ms,
            "fetch_ms": self.fetch_ms,
            "extraction_ms": self.extraction_ms,
            "fetched_bytes": float(self.fetched_bytes),
        }


__all__ = ["WebPageReaderBackend", "extract_page", "validate_public_url"]
