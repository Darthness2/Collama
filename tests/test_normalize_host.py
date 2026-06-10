"""ollama_client._normalize_host — fixes up user-typed hosts.

This is the bit that turns 'localhost' / '0.0.0.0' / 'foo:8080' into a
URL that requests can actually connect to. Bugs here surface as silent
'connection refused' so it's worth nailing down.
"""
from __future__ import annotations

import unittest

from collama.ollama_client import _normalize_host


class NormalizeHostTests(unittest.TestCase):
    def test_empty_input_returns_default(self):
        self.assertEqual(_normalize_host(""), "http://localhost:11434")

    def test_none_input_returns_default(self):
        self.assertEqual(_normalize_host(None), "http://localhost:11434")  # type: ignore[arg-type]

    def test_bare_hostname_gets_scheme_and_default_port(self):
        self.assertEqual(_normalize_host("localhost"), "http://localhost:11434")

    def test_hostname_with_port_keeps_port(self):
        self.assertEqual(_normalize_host("localhost:9000"), "http://localhost:9000")

    def test_wildcard_ipv4_remaps_to_loopback(self):
        """0.0.0.0 is a bind address, not a connect target — on macOS/Windows
        connecting to it fails. Must be rewritten to 127.0.0.1."""
        self.assertEqual(_normalize_host("0.0.0.0"), "http://127.0.0.1:11434")

    def test_wildcard_ipv4_with_explicit_port_remaps_loopback_keeps_port(self):
        self.assertEqual(_normalize_host("0.0.0.0:11434"),
                         "http://127.0.0.1:11434")

    def test_wildcard_with_scheme_remaps_loopback(self):
        self.assertEqual(_normalize_host("http://0.0.0.0:11434"),
                         "http://127.0.0.1:11434")

    def test_https_remote_preserved(self):
        self.assertEqual(_normalize_host("https://ollama.example.com:443"),
                         "https://ollama.example.com:443")

    def test_url_with_custom_path_left_alone(self):
        """If the user typed a path, assume they know what they're doing."""
        url = _normalize_host("http://example.com/ollama")
        self.assertEqual(url, "http://example.com/ollama")

    def test_trailing_slash_stripped(self):
        self.assertEqual(_normalize_host("http://localhost:11434/"),
                         "http://localhost:11434")

    def test_scheme_only_input_gets_default_port(self):
        self.assertEqual(_normalize_host("http://localhost"),
                         "http://localhost:11434")


if __name__ == "__main__":
    unittest.main()
